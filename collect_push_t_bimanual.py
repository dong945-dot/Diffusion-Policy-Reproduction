import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class NaturalBimanualPusher:
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

        # 【核心约束姿态】：自然微曲、手肘外展下垂，坚决禁止手肘冲天
        self.q_nominal = np.zeros(self.model.nu)
        if self.model.nu >= 14:
            self.q_nominal[:7] = [0.0, 0.45, 0.0, -1.2, 0.0, 0.75, 0.0]
            self.q_nominal[7:14] = [0.0, -0.45, 0.0, -1.2, 0.0, 0.75, 0.0]

    def solve_ik(self, target_l, target_r, steps=5):
        """带强手肘下压约束的 IK，彻底根除手肘耸起翻飞"""
        for _ in range(steps):
            curr_l = self.data.xpos[self.left_ee_id]
            curr_r = self.data.xpos[self.right_ee_id]

            err = np.clip(np.concatenate([target_l - curr_l, target_r - curr_r]), -0.025, 0.025)
            if np.linalg.norm(err) < 0.0015:
                break

            jacp_l, jacp_r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr, self.left_ee_id)
            mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr, self.right_ee_id)

            J = np.vstack([jacp_l, jacp_r])
            reg = 0.004 * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg)
            dq_main = J_pinv @ err

            # 零空间自构形控制：强行拉住手肘，维持端正姿态
            I = np.eye(self.model.nv)
            dq_null = np.zeros(self.model.nv)
            dq_null[:self.model.nu] = 0.5 * (self.q_nominal - self.data.qpos[:self.model.nu])

            dq = dq_main + (I - J_pinv @ J) @ dq_null
            dq[self.model.nu:] = 0.0
            dq = np.clip(dq, -0.035, 0.035)

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)

        return self.data.qpos[:self.model.nu].copy()

    def reset_env(self):
        mujoco.mj_resetData(self.model, self.data)

        # 黄金可达区
        t_x = np.random.uniform(0.22, 0.24)
        t_y = np.random.uniform(-0.02, 0.02)
        t_yaw = np.random.uniform(-np.pi/6, np.pi/6)

        g_x = np.random.uniform(0.27, 0.29)
        g_y = np.random.uniform(-0.015, 0.015)
        g_yaw = 0.0

        # 设置 T 块位姿
        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()
        t_quat_wxyz = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]]
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.195]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = t_quat_wxyz

        # 设置目标
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        self.model.body_quat[goal_body_id] = [1.0, 0.0, 0.0, 0.0]

        # 机械臂初始就位（双手开度拉大到 10.5cm，彻底避免夹爪打架）
        push_dir = np.array([g_x - t_x, g_y - t_y])
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])

        p_center = np.array([t_x, t_y]) - 0.025 * push_dir
        # 单侧外扩 5.3cm，双爪净间距 10.6cm
        init_l = np.array([p_center[0] + 0.053 * push_perp[0], p_center[1] + 0.053 * push_perp[1], 0.193])
        init_r = np.array([p_center[0] - 0.053 * push_perp[0], p_center[1] - 0.053 * push_perp[1], 0.193])

        # 初始赋姿态并平滑就位
        self.data.qpos[:len(self.q_nominal)] = self.q_nominal
        for _ in range(15):
            self.solve_ik(init_l, init_r, steps=5)

        mujoco.mj_forward(self.model, self.data)

        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect(self, num_episodes=10, max_steps=100):
        obs_imgs, obs_states, actions, qposes = [], [], [], []
        t_inits, g_inits, ep_ends = [], [], []
        total_steps = 0

        print(f"\n🚀 开始执行【防打架 + 沉肩下推】高质量示教采集，共 {num_episodes} 条...")

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

                diff_pos = g_pos[:2] - t_pos[:2]
                dist = np.linalg.norm(diff_pos)
                push_dir = diff_pos / (dist + 1e-6)
                push_perp = np.array([-push_dir[1], push_dir[0]])

                yaw_err = (g_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi
                yaw_err = np.clip(yaw_err, -0.25, 0.25)

                # 💡【终点柔和刹车】：越接近目标推进步幅越平缓，消除终点晃荡
                step_advance = 0.003 if dist > 0.02 else (0.001 * (dist / 0.02))
                push_center = t_pos[:2] + step_advance * push_dir

                # 宽距差速护航（间距 10.6cm）
                left_xy = push_center + 0.053 * push_perp - 0.04 * yaw_err * push_dir
                right_xy = push_center - 0.053 * push_perp + 0.04 * yaw_err * push_dir

                target_l = np.array([left_xy[0], left_xy[1], 0.193])
                target_r = np.array([right_xy[0], right_xy[1], 0.193])
                action = np.concatenate([target_l, target_r])

                target_q = self.solve_ik(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

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

                # 判定精准达标
                if dist < 0.016 and abs(yaw_err) < np.deg2rad(8):
                    print(f"▶ Episode [{ep+1}/{num_episodes}]: ✔ 完美推入！耗时 {step+1} 步平稳收官！")
                    break
            else:
                print(f"▶ Episode [{ep+1}/{num_episodes}]: 接近目标 (最终距离 {dist*100:.1f}cm)")

            ep_ends.append(total_steps)

        # 写入 Zarr 数据集
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

        print(f"\n全部采集完毕！已真实保存至: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")
    collector = NaturalBimanualPusher(xml_path=xml_path)
    collector.collect(num_episodes=10, max_steps=100)
