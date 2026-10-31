import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class RobustBimanualPushCollector:
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

        # 舒适初始下探姿态（前伸下垂，绝不向后翻）
        self.q_home = np.zeros(self.model.nu)
        if self.model.nu >= 14:
            self.q_home[:7] = [0.0, 0.4, 0.0, -1.2, 0.0, 0.8, 0.0]
            self.q_home[7:14] = [0.0, -0.4, 0.0, -1.2, 0.0, 0.8, 0.0]

    def solve_ik(self, target_l, target_r):
        """带极强零空间名义姿态约束的 IK，死死按住手肘防止向后甩"""
        for _ in range(4):
            curr_l = self.data.xpos[self.left_ee_id]
            curr_r = self.data.xpos[self.right_ee_id]

            err = np.clip(np.concatenate([target_l - curr_l, target_r - curr_r]), -0.025, 0.025)
            if np.linalg.norm(err) < 0.002:
                break

            jacp_l, jacp_r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr, self.left_ee_id)
            mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr, self.right_ee_id)

            J = np.vstack([jacp_l, jacp_r])
            reg = 0.005 * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg)

            dq_main = J_pinv @ err

            # 强行拉回前伸正常姿态
            dq_null = np.zeros(self.model.nv)
            dq_null[:self.model.nu] = 0.6 * (self.q_home - self.data.qpos[:self.model.nu])

            dq = dq_main + (np.eye(self.model.nv) - J_pinv @ J) @ dq_null
            dq[self.model.nu:] = 0.0
            dq = np.clip(dq, -0.03, 0.03)

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)

        return self.data.qpos[:self.model.nu].copy()

    def get_funnel_targets(self, t_pos, t_yaw, goal_pos, goal_yaw, push_z=0.193):
        """漏斗铲双臂规划器：左手永远在左侧，右手永远在右侧，绝不交叉"""
        diff_pos = goal_pos[:2] - t_pos[:2]
        dist = np.linalg.norm(diff_pos)
        push_dir = diff_pos / (dist + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])  # 垂直于推行方向的侧向向量

        # 计算 T 块后方的支撑点
        behind_center = t_pos[:2] - 0.045 * push_dir

        # 两点分别在推进向量的左侧与右侧
        pt1 = behind_center + 0.045 * push_perp
        pt2 = behind_center - 0.045 * push_perp

        # 【核心禁令：严格按世界 Y 坐标分配，大 Y 归左手，小 Y 归右手】
        if pt1[1] >= pt2[1]:
            tgt_l = np.array([pt1[0], pt1[1], push_z])
            tgt_r = np.array([pt2[0], pt2[1], push_z])
        else:
            tgt_l = np.array([pt2[0], pt2[1], push_z])
            tgt_r = np.array([pt1[0], pt1[1], push_z])

        return tgt_l, tgt_r

    def reset_env(self):
        mujoco.mj_resetData(self.model, self.data)

        # 随机位姿（严格限制在前方 40cm 工作区内）
        t_x = np.random.uniform(0.38, 0.42)
        t_y = np.random.uniform(-0.03, 0.03)
        t_yaw = np.random.uniform(-np.pi/4, np.pi/4)

        g_x = np.random.uniform(0.40, 0.43)
        g_y = np.random.uniform(-0.02, 0.02)
        g_yaw = 0.0  # 目标朝向放正，降低初期复杂度

        # 设置 T 块与目标
        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()
        t_quat_wxyz = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]]
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.195]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = t_quat_wxyz

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        self.model.body_quat[goal_body_id] = [1.0, 0.0, 0.0, 0.0]

        # 强行将机械臂复位到前伸姿态
        self.data.qpos[:len(self.q_home)] = self.q_home
        mujoco.mj_forward(self.model, self.data)

        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect(self, num_episodes=10, max_steps=180):
        obs_imgs, obs_states, actions, qposes = [], [], [], []
        t_inits, g_inits, ep_ends = [], [], []
        total_steps = 0

        print(f"\n🚀 开始执行【防交叉、强手肘约束】高质量采集，共 {num_episodes} 条...")

        for ep in range(num_episodes):
            t_pos, t_yaw, g_pos, g_yaw, t_7d, g_7d = self.reset_env()
            t_inits.append(t_7d)
            g_inits.append(g_7d)

            # 两手当前设定点从当前真实位置开始渐进平滑
            curr_tgt_l = self.data.xpos[self.left_ee_id].copy()
            curr_tgt_r = self.data.xpos[self.right_ee_id].copy()

            for step in range(max_steps):
                self.renderer.update_scene(self.data, camera=self.camera_name)
                img = self.renderer.render()

                l_pos = self.data.xpos[self.left_ee_id].copy()
                r_pos = self.data.xpos[self.right_ee_id].copy()
                ee_state = np.concatenate([l_pos, r_pos])

                # 闭环计算漏斗铲目标点
                desired_l, desired_r = self.get_funnel_targets(t_pos, t_yaw, g_pos, g_yaw)

                # 限制步长最大 3mm，防止猛冲
                curr_tgt_l += np.clip(desired_l - curr_tgt_l, -0.003, 0.003)
                curr_tgt_r += np.clip(desired_r - curr_tgt_r, -0.003, 0.003)

                action = np.concatenate([curr_tgt_l, curr_tgt_r])

                # IK 驱动
                target_q = self.solve_ik(curr_tgt_l, curr_tgt_r)
                self.data.ctrl[:len(target_q)] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

                # 记录真实的关节角度（最关键的一步！）
                current_arm_qpos = self.data.qpos[:self.model.nu].copy()

                t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
                t_pos = self.data.qpos[t_joint_id:t_joint_id+3]

                obs_imgs.append(img)
                obs_states.append(ee_state)
                actions.append(action)
                qposes.append(current_arm_qpos)
                total_steps += 1

                # 成功收敛
                if np.linalg.norm(g_pos[:2] - t_pos[:2]) < 0.018:
                    break

            ep_ends.append(total_steps)
            print(f"✔ Episode [{ep+1}/{num_episodes}] 录制完毕 (耗时 {step+1} 步)")

        # 写入 Zarr 数据集（包含了真实的 joint_qpos）
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

        print(f"\n全部 10 条数据已真实保存至: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")
    collector = RobustBimanualPushCollector(xml_path=xml_path)
    collector.collect(num_episodes=10, max_steps=180)
