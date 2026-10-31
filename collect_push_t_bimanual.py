import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class RealWorldBimanualPusher:
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

        # 左右手各自名义姿态（自然下垂前伸，手肘外扩）
        self.q_left_home = np.array([0.0, 0.4, 0.0, -1.1, 0.0, 0.7, 0.0])
        self.q_right_home = np.array([0.0, -0.4, 0.0, -1.1, 0.0, 0.7, 0.0])

    def solve_single_arm_ik(self, ee_id, target_pos, joint_start_idx, q_home, steps=4):
        """【核心解耦】：单臂独立求解 IK，杜绝两臂耦合导致一臂飞天"""
        for _ in range(steps):
            curr_pos = self.data.xpos[ee_id]
            err = np.clip(target_pos - curr_pos, -0.025, 0.025)
            if np.linalg.norm(err) < 0.0015:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, ee_id)

            # 只提取对应机械臂的 7 个关节列
            J_arm = jacp[:, joint_start_idx : joint_start_idx + 7]
            reg = 0.002 * np.eye(3)
            J_pinv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + reg)

            dq_main = J_pinv @ err

            # 姿态下压约束
            q_curr = self.data.qpos[joint_start_idx : joint_start_idx + 7]
            dq_null = 0.4 * (q_home - q_curr)
            dq = dq_main + (np.eye(7) - J_pinv @ J_arm) @ dq_null
            dq = np.clip(dq, -0.035, 0.035)

            # 积分本机械臂的 7 个关节
            self.data.qpos[joint_start_idx : joint_start_idx + 7] += dq
            mujoco.mj_kinematics(self.model, self.data)

    def solve_bimanual(self, target_l, target_r):
        """左右两臂互不干扰独立逆解"""
        self.solve_single_arm_ik(self.left_ee_id, target_l, joint_start_idx=0, q_home=self.q_left_home)
        self.solve_single_arm_ik(self.right_ee_id, target_r, joint_start_idx=7, q_home=self.q_right_home)
        return self.data.qpos[:14].copy()

    def reset_env(self):
        mujoco.mj_resetData(self.model, self.data)

        # 舒适推挤区域：X = 0.27 ~ 0.33
        t_x = np.random.uniform(0.27, 0.29)
        t_y = np.random.uniform(-0.02, 0.02)
        t_yaw = np.random.uniform(-np.pi/6, np.pi/6)

        g_x = np.random.uniform(0.33, 0.35)
        g_y = np.random.uniform(-0.015, 0.015)
        g_yaw = 0.0

        # 设置 T 块位姿
        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()
        t_quat_wxyz = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]]
        # 初始稍微悬空 5 毫米，防止初始穿模弹飞
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.200]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = t_quat_wxyz

        # 设置目标虚影
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        self.model.body_quat[goal_body_id] = [1.0, 0.0, 0.0, 0.0]

        # 机械臂初始姿态注入
        self.data.qpos[:7] = self.q_left_home
        self.data.qpos[7:14] = self.q_right_home

        # 【核心物理步骤】：前向沉降 15 步，让 T 块平稳受重力沉落桌面上，释放接触应力！
        for _ in range(15):
            mujoco.mj_step(self.model, self.data)

        # 沉降完成后，读取真正静止的真实物理位姿
        t_pos_settled = self.data.qpos[t_joint_id:t_joint_id+3].copy()

        # 机械臂贴合就位：双手从 T 块后方两侧轻轻贴近
        push_dir = np.array([g_x - t_pos_settled[0], g_y - t_pos_settled[1]])
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])

        p_center = t_pos_settled[:2] - 0.02 * push_dir
        init_l = np.array([p_center[0] + 0.052 * push_perp[0], p_center[1] + 0.052 * push_perp[1], 0.193])
        init_r = np.array([p_center[0] - 0.052 * push_perp[0], p_center[1] - 0.052 * push_perp[1], 0.193])

        for _ in range(20):
            self.solve_bimanual(init_l, init_r)

        mujoco.mj_forward(self.model, self.data)

        t_init_7d = np.array([t_pos_settled[0], t_pos_settled[1], t_pos_settled[2]] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return t_pos_settled, t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect(self, num_episodes=10, max_steps=100):
        obs_imgs, obs_states, actions, qposes = [], [], [], []
        t_inits, g_inits, ep_ends = [], [], []
        total_steps = 0

        print(f"\n🚀 开始执行【解耦双臂 + 真实重力沉降】物理推进，共 {num_episodes} 条...")

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

                # 推进步进
                step_advance = 0.003 if dist > 0.02 else (0.001 * (dist / 0.02))
                push_center = t_pos[:2] + step_advance * push_dir

                left_xy = push_center + 0.052 * push_perp - 0.035 * yaw_err * push_dir
                right_xy = push_center - 0.052 * push_perp + 0.035 * yaw_err * push_dir

                target_l = np.array([left_xy[0], left_xy[1], 0.193])
                target_r = np.array([right_xy[0], right_xy[1], 0.193])
                action = np.concatenate([target_l, target_r])

                target_q = self.solve_bimanual(target_l, target_r)
                self.data.ctrl[:14] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

                current_arm_qpos = self.data.qpos[:14].copy()
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

                if dist < 0.016 and abs(yaw_err) < np.deg2rad(8):
                    print(f"▶ Episode [{ep+1}/{num_episodes}]: ✔ 真实平稳推入！用时 {step+1} 步！")
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
        data_grp.create_dataset('joint_qpos', data=np.array(qposes, dtype=np.float32), chunks=(1000, 14))
        data_grp.create_dataset('t_init_pose', data=np.array(t_inits, dtype=np.float32))
        data_grp.create_dataset('goal_pose', data=np.array(g_inits, dtype=np.float32))
        meta_grp.create_dataset('episode_ends', data=np.array(ep_ends, dtype=np.int64))

        print(f"\n全部采集完毕！已真实保存至: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")
    collector = RealWorldBimanualPusher(xml_path=xml_path)
    collector.collect(num_episodes=10, max_steps=100)
