import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class DifferentialBimanualPusher:
    def __init__(self, xml_path="openarm_pusht.xml", zarr_path="push_t_demo.zarr"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.left_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7")
        self.right_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7")
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand")
            self.right_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_hand")

        self.camera_name = "top_cam"
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.zarr_path = zarr_path

    def solve_ik(self, target_l, target_r, steps=5):
        """精准逆解，强行压制手腕垂直朝下"""
        for _ in range(steps):
            curr_l = self.data.xpos[self.left_ee_id]
            curr_r = self.data.xpos[self.right_ee_id]

            err = np.clip(np.concatenate([target_l - curr_l, target_r - curr_r]), -0.03, 0.03)
            if np.linalg.norm(err) < 0.0015:
                break

            jacp_l, jacp_r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr, self.left_ee_id)
            mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr, self.right_ee_id)

            J = np.vstack([jacp_l, jacp_r])
            reg = 0.004 * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg)
            dq = J_pinv @ err
            dq[self.model.nu:] = 0.0
            dq = np.clip(dq, -0.04, 0.04)

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)

        return self.data.qpos[:self.model.nu].copy()

    def reset_env(self):
        mujoco.mj_resetData(self.model, self.data)

        # 1. 随机 T 块起点与目标
        t_x = np.random.uniform(0.37, 0.41)
        t_y = np.random.uniform(-0.03, 0.03)
        t_yaw = np.random.uniform(-np.pi/4, np.pi/4)

        g_x = np.random.uniform(0.42, 0.45)
        g_y = np.random.uniform(-0.02, 0.02)
        g_yaw = 0.0

        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()
        t_quat_wxyz = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]]
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.195]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = t_quat_wxyz

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        self.model.body_quat[goal_body_id] = [1.0, 0.0, 0.0, 0.0]

        # 2. 【核心优化】：消除大鹏展翅！在重置瞬间直接将手臂 IK 解算到 T 块后方就位！
        push_dir = np.array([g_x - t_x, g_y - t_y])
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])

        # 初始紧贴 T 块后方 1.5cm 处
        p_init_center = np.array([t_x, t_y]) - 0.02 * push_dir
        init_l = np.array([p_init_center[0] + 0.04 * push_perp[0], p_init_center[1] + 0.04 * push_perp[1], 0.194])
        init_r = np.array([p_init_center[0] - 0.04 * push_perp[0], p_init_center[1] - 0.04 * push_perp[1], 0.194])

        # 多步迭代直接将手臂放到桌前就位
        for _ in range(15):
            self.solve_ik(init_l, init_r, steps=5)

        mujoco.mj_forward(self.model, self.data)

        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect(self, num_episodes=10, max_steps=120):
        obs_imgs, obs_states, actions, qposes = [], [], [], []
        t_inits, g_inits, ep_ends = [], [], []
        total_steps = 0

        print(f"\n🚀 开始执行【差速推进推土铲】示教采集，共 {num_episodes} 条...")

        for ep in range(num_episodes):
            t_pos, t_yaw, g_pos, g_yaw, t_7d, g_7d = self.reset_env()
            t_inits.append(t_7d)
            g_inits.append(g_7d)

            for step in range(max_steps):
                self.renderer.update_scene(self.data, camera=self.camera_name)
                img = self.renderer.render()

                l_pos = self.data.xpos[self.left_ee_id].copy()
                r_pos = self.data.xpos[self.right_ee_id].copy()
                ee_state = np.concatenate([l_pos, r_pos])

                # -------------------------------------------------------------
                # 差速推土铲核心控制器
                # -------------------------------------------------------------
                diff_pos = g_pos[:2] - t_pos[:2]
                dist = np.linalg.norm(diff_pos)
                push_dir = diff_pos / (dist + 1e-6)
                push_perp = np.array([-push_dir[1], push_dir[0]])

                # 角度误差
                yaw_err = (g_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi
                yaw_err = np.clip(yaw_err, -0.4, 0.4)

                # 接触推进中心：直接咬住 T 块质心向前推入！
                push_center = t_pos[:2] + 0.003 * push_dir

                # 左右手差速纠偏（关键：边推边校正旋转角）
                left_xy = push_center + 0.038 * push_perp - 0.06 * yaw_err * push_dir
                right_xy = push_center - 0.038 * push_perp + 0.06 * yaw_err * push_dir

                target_l = np.array([left_xy[0], left_xy[1], 0.193])
                target_r = np.array([right_xy[0], right_xy[1], 0.193])

                action = np.concatenate([target_l, target_r])

                # 驱动执行
                target_q = self.solve_ik(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

                # 真实关节与 T 块更新
                current_arm_qpos = self.data.qpos[:self.model.nu].copy()
                t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
                t_pos = self.data.qpos[t_joint_id:t_joint_id+3]
                q_wxyz = self.data.qpos[t_joint_id+3:t_joint_id+7]
                r_mat = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                t_yaw = r_mat.as_euler('zyx')[0]

                obs_imgs.append(img)
                obs_states.append(ee_state)
                actions.append(action)
                qposes.append(current_arm_qpos)
                total_steps += 1

                # 成功进洞判据（距离小于 1.8cm 且 角度对齐小于 10 度）
                if dist < 0.018 and abs(yaw_err) < np.deg2rad(10):
                    print(f"▶ Episode [{ep+1}/{num_episodes}]: ✔ 完美推入目标！仅耗时 {step+1} 步！")
                    break
            else:
                print(f"▶ Episode [{ep+1}/{num_episodes}]: 接近目标 (最终距离 {dist*100:.1f}cm)")

            ep_ends.append(total_steps)

        # 写入 Zarr
        root = zarr.open(self.zarr_path, mode='w')
        data_grp = root.create_group('data')
        meta_grp = root.create_group('meta')
        data_grp.create_dataset('img', data=np.array(obs_imgs, dtype=np.uint8), chunks=(100, 128, 128, 3))
        data_grp.create_dataset('state', data=np.array(obs_states, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('action', data=np.array(actions, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('joint_qpos', data=np.array(qposes, dtype=np.float32), chunks=(1000, self.model.nu))
        data_grp.create_dataset('t_init_pose', data=np.array(t_inits, dtype=np.float32))
        data_grp.create_dataset('goal_pose', data=np.array(g_inits, dtype=np.float32))
        meta_grp.create_dataset('episode_ends', data=np.array(ep_ends, dtype=np.int64))

        print(f"\n全部采集完成！已存入: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")
    collector = DifferentialBimanualPusher(xml_path=xml_path)
    collector.collect(num_episodes=10, max_steps=120)
