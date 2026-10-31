import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# 1. 3D 逆运动学 (IK) 求解器
# ==============================================================================
class BimanualIKSolver:
    def __init__(self, model, data, left_ee_name, right_ee_name, damping=1e-3):
        self.model = model
        self.data = data
        self.damping = damping
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)
        
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_ee_name)
            self.use_site = True
        else:
            self.use_site = False

    def solve(self, left_target_pos, right_target_pos, max_steps=5):
        for _ in range(max_steps):
            if self.use_site:
                curr_l = self.data.site_xpos[self.left_ee_id]
                curr_r = self.data.site_xpos[self.right_ee_id]
            else:
                curr_l = self.data.xpos[self.left_ee_id]
                curr_r = self.data.xpos[self.right_ee_id]

            err_l = left_target_pos - curr_l
            err_r = right_target_pos - curr_r
            error = np.concatenate([err_l, err_r])

            if np.linalg.norm(error) < 0.003:
                break

            jacp_l = np.zeros((3, self.model.nv))
            jacp_r = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            
            if self.use_site:
                mujoco.mj_jacSite(self.model, self.data, jacp_l, jacr, self.left_ee_id)
                mujoco.mj_jacSite(self.model, self.data, jacp_r, jacr, self.right_ee_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr, self.left_ee_id)
                mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr, self.right_ee_id)

            J = np.vstack([jacp_l, jacp_r])
            reg = self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(J @ J.T + reg, error)
            
            dq = np.clip(dq, -0.1, 0.1)
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)
            
        return self.data.qpos[:self.model.nu].copy()


# ==============================================================================
# 2. 推 T 启发式专家策略
# ==============================================================================
class BimanualPushTExpert:
    def __init__(self, push_height=0.195, retract_height=0.25):
        self.push_z = push_height
        self.retract_z = retract_height

    def get_action(self, t_pos, t_yaw, goal_pos, goal_yaw):
        diff_yaw = (goal_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pos = goal_pos[:2] - t_pos[:2]
        dist = np.linalg.norm(diff_pos)

        # 阶段一：调角度
        if abs(diff_yaw) > np.deg2rad(15):
            t_dir = np.array([np.cos(t_yaw), np.sin(t_yaw)])
            t_perp = np.array([-np.sin(t_yaw), np.cos(t_yaw)])
            head_left = t_pos[:2] + 0.08 * t_dir + 0.03 * t_perp
            head_right = t_pos[:2] - 0.08 * t_dir + 0.03 * t_perp
            
            if diff_yaw > 0:
                left_target = np.array([head_left[0] - 0.02 * t_perp[0], head_left[1] - 0.02 * t_perp[1], self.push_z])
                right_target = np.array([head_right[0] + 0.02 * t_perp[0], head_right[1] + 0.02 * t_perp[1], self.push_z])
            else:
                left_target = np.array([head_left[0] + 0.02 * t_perp[0], head_left[1] + 0.02 * t_perp[1], self.push_z])
                right_target = np.array([head_right[0] - 0.02 * t_perp[0], head_right[1] - 0.02 * t_perp[1], self.push_z])
            return left_target, right_target

        # 阶段二：协同平移
        push_dir = diff_pos / (dist + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])
        behind_center = t_pos[:2] - 0.06 * push_dir
        
        target_left_2d = behind_center + 0.045 * push_perp + 0.02 * push_dir
        target_right_2d = behind_center - 0.045 * push_perp + 0.02 * push_dir

        left_target = np.array([target_left_2d[0], target_left_2d[1], self.push_z])
        right_target = np.array([target_right_2d[0], target_right_2d[1], self.push_z])
        return left_target, right_target


# ==============================================================================
# 3. 数据采集与存储引擎
# ==============================================================================
class DataCollector:
    def __init__(self, xml_path, zarr_save_path="push_t_demo.zarr"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.left_ee_name = "openarm_left_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7") != -1 else "left_hand"
        self.right_ee_name = "openarm_right_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7") != -1 else "right_hand"
        
        self.ik_solver = BimanualIKSolver(self.model, self.data, self.left_ee_name, self.right_ee_name)
        self.expert = BimanualPushTExpert()
        
        self.camera_name = "top_cam"
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.zarr_path = zarr_save_path

    def reset_environment(self):
        """【修改点 1】初始化环境并返回包含四元数的 7 维初始位姿数组"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 采样 T 块
        t_x = np.random.uniform(0.36, 0.46)
        t_y = np.random.uniform(-0.08, 0.08)
        t_yaw = np.random.uniform(-np.pi, np.pi)
        
        # 采样 目标 Target
        g_x = np.random.uniform(0.38, 0.44)
        g_y = np.random.uniform(-0.06, 0.06)
        g_yaw = np.random.uniform(-np.pi/3, np.pi/3)

        # 设置 T 块位姿 (MuJoCo 四元数格式为 [w, x, y, z])
        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()  # scipy 输出为 [x, y, z, w]
        t_quat_wxyz = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]]
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.195]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = t_quat_wxyz

        # 设置目标虚影 (静止体)
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        quat_g = R.from_euler('z', g_yaw).as_quat()
        g_quat_wxyz = [quat_g[3], quat_g[0], quat_g[1], quat_g[2]]
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        self.model.body_quat[goal_body_id] = g_quat_wxyz

        mujoco.mj_forward(self.model, self.data)

        # 封装为 7 维数组 [x, y, z, w, qx, qy, qz]
        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181] + g_quat_wxyz, dtype=np.float32)

        return (
            np.array([t_x, t_y, 0.195]), 
            t_yaw, 
            np.array([g_x, g_y, 0.181]), 
            g_yaw, 
            t_init_7d, 
            goal_init_7d
        )

    def collect_dataset(self, num_episodes=10, max_steps=180):
        root = zarr.open(self.zarr_path, mode='w')
        data_grp = root.create_group('data')
        meta_grp = root.create_group('meta')

        obs_img_list = []
        obs_state_list = []
        action_list = []
        episode_ends = []
        
        # 【修改点 2】创建用来存储每条轨迹初始状态的列表
        t_init_list = []
        goal_init_list = []
        
        total_steps = 0
        print(f"开始采集 {num_episodes} 条双臂 Push-T 轨迹（含初始位姿保存）...")

        for ep in range(num_episodes):
            t_pos, t_yaw, g_pos, g_yaw, t_init_7d, goal_init_7d = self.reset_environment()
            
            # 【修改点 3】记录当前轨迹的初始 T 块和目标位姿
            t_init_list.append(t_init_7d)
            goal_init_list.append(goal_init_7d)

            for step in range(max_steps):
                self.renderer.update_scene(self.data, camera=self.camera_name)
                img = self.renderer.render()

                l_pos = self.data.xpos[self.ik_solver.left_ee_id].copy()
                r_pos = self.data.xpos[self.ik_solver.right_ee_id].copy()
                agent_state = np.concatenate([l_pos, r_pos])

                target_l, target_r = self.expert.get_action(t_pos, t_yaw, g_pos, g_yaw)
                action = np.concatenate([target_l, target_r])

                target_q = self.ik_solver.solve(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q
                
                for _ in range(5):
                    mujoco.mj_step(self.model, self.data)

                t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
                t_pos = self.data.qpos[t_joint_id:t_joint_id+3]
                q_wxyz = self.data.qpos[t_joint_id+3:t_joint_id+7]
                r_mat = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                t_yaw = r_mat.as_euler('zyx')[0]

                obs_img_list.append(img)
                obs_state_list.append(agent_state)
                action_list.append(action)
                total_steps += 1

                if np.linalg.norm(g_pos[:2] - t_pos[:2]) < 0.015 and abs((g_yaw - t_yaw + np.pi) % (2*np.pi) - np.pi) < np.deg2rad(10):
                    print(f"Episode {ep+1} 成功达成目标，耗时 {step+1} 步！")
                    break

            episode_ends.append(total_steps)
            print(f"完成进度: [{ep+1}/{num_episodes}]")

        # 转换为 numpy 格式
        img_np = np.array(obs_img_list, dtype=np.uint8)
        state_np = np.array(obs_state_list, dtype=np.float32)
        action_np = np.array(action_list, dtype=np.float32)
        ends_np = np.array(episode_ends, dtype=np.int64)
        t_init_np = np.array(t_init_list, dtype=np.float32)
        goal_init_np = np.array(goal_init_list, dtype=np.float32)

        # 【修改点 4】写入 Zarr 数据集
        data_grp.create_dataset('img', data=img_np, chunks=(100, 128, 128, 3))
        data_grp.create_dataset('state', data=state_np, chunks=(1000, 6))
        data_grp.create_dataset('action', data=action_np, chunks=(1000, 6))
        data_grp.create_dataset('t_init_pose', data=t_init_np)  # 存储 T 块初始位姿 (N_episodes, 7)
        data_grp.create_dataset('goal_pose', data=goal_init_np)  # 存储 目标区位姿 (N_episodes, 7)
        meta_grp.create_dataset('episode_ends', data=ends_np)

        print(f"\n全部采集完毕！已完整保存 T 块初始位姿至：{os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_push_t.xml")
    
    collector = DataCollector(xml_path=xml_path)
    collector.collect_dataset(num_episodes=10, max_steps=180)
