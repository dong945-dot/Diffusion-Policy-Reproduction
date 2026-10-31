import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# 1. 3D 逆运动学 (IK) 求解器：将双臂笛卡尔路径反解为关节控制目标
# ==============================================================================
class BimanualIKSolver:
    def __init__(self, model, data, left_ee_name, right_ee_name, damping=1e-3):
        self.model = model
        self.data = data
        self.damping = damping
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)
        
        # 兼容性兜底：若末端命名为site则查找site
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_ee_name)
            self.use_site = True
        else:
            self.use_site = False

    def solve(self, left_target_pos, right_target_pos, max_steps=5):
        """利用雅可比阻尼最小二乘法 (Damped Least Squares) 迭代反解双臂末端"""
        for _ in range(max_steps):
            # 获取当前两末端真实世界位置
            if self.use_site:
                curr_l = self.data.site_xpos[self.left_ee_id]
                curr_r = self.data.site_xpos[self.right_ee_id]
            else:
                curr_l = self.data.xpos[self.left_ee_id]
                curr_r = self.data.xpos[self.right_ee_id]

            err_l = left_target_pos - curr_l
            err_r = right_target_pos - curr_r
            error = np.concatenate([err_l, err_r])  # 6维位置误差 (3 + 3)

            if np.linalg.norm(error) < 0.003:
                break

            # 计算两机械臂的雅可比矩阵
            jacp_l = np.zeros((3, self.model.nv))
            jacp_r = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))  # 忽略旋转约束，降低奇异点风险
            
            if self.use_site:
                mujoco.mj_jacSite(self.model, self.data, jacp_l, jacr, self.left_ee_id)
                mujoco.mj_jacSite(self.model, self.data, jacp_r, jacr, self.right_ee_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr, self.left_ee_id)
                mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr, self.right_ee_id)

            J = np.vstack([jacp_l, jacp_r])  # (6, nv)
            
            # 阻尼最小二乘：dq = J^T (J J^T + lambda^2 I)^-1 * err
            reg = self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(J @ J.T + reg, error)
            
            # 单步更新并限制最大单步步长，防止关节剧烈抽搐
            dq = np.clip(dq, -0.1, 0.1)
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)
            
        return self.data.qpos[:self.model.nu].copy()


# ==============================================================================
# 2. 推 T 启发式专家策略 (Bimanual Push-T Expert Heuristic)
# ==============================================================================
class BimanualPushTExpert:
    def __init__(self, push_height=0.195, retract_height=0.25):
        self.push_z = push_height        # 触碰推动物体的推挤工作高度
        self.retract_z = retract_height  # 抬手过渡防碰撞巡航高度

    def get_action(self, t_pos, t_yaw, goal_pos, goal_yaw):
        """
        输入当前 T 块与目标 T 块的状态，输出左手和右手的笛卡尔 3D 期望目标
        """
        diff_yaw = (goal_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pos = goal_pos[:2] - t_pos[:2]
        dist = np.linalg.norm(diff_pos)

        # ----------------------------------------------------
        # 阶段一：姿态调整 (角度误差大于 15 度)
        # 利用左右两臂反向推动 T 的两端施加转矩 (Couple Force)
        # ----------------------------------------------------
        if abs(diff_yaw) > np.deg2rad(15):
            t_dir = np.array([np.cos(t_yaw), np.sin(t_yaw)])
            t_perp = np.array([-np.sin(t_yaw), np.cos(t_yaw)])
            
            # 头部两侧接触点
            head_left = t_pos[:2] + 0.08 * t_dir + 0.03 * t_perp
            head_right = t_pos[:2] - 0.08 * t_dir + 0.03 * t_perp
            
            if diff_yaw > 0:  # 逆时针转
                left_target = np.array([head_left[0] - 0.02 * t_perp[0], head_left[1] - 0.02 * t_perp[1], self.push_z])
                right_target = np.array([head_right[0] + 0.02 * t_perp[0], head_right[1] + 0.02 * t_perp[1], self.push_z])
            else:             # 顺时针转
                left_target = np.array([head_left[0] + 0.02 * t_perp[0], head_left[1] + 0.02 * t_perp[1], self.push_z])
                right_target = np.array([head_right[0] - 0.02 * t_perp[0], head_right[1] - 0.02 * t_perp[1], self.push_z])
            return left_target, right_target

        # ----------------------------------------------------
        # 阶段二：协同平移 (角度已大致对准，双臂一左一右对称协同推进)
        # ----------------------------------------------------
        push_dir = diff_pos / (dist + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])  # 垂直平移方向的法向量
        
        # 站在物体背后 6cm 处
        behind_center = t_pos[:2] - 0.06 * push_dir
        
        # 双手分开 7cm，形成夹角向前拱卫推进
        target_left_2d = behind_center + 0.045 * push_perp + 0.02 * push_dir
        target_right_2d = behind_center - 0.045 * push_perp + 0.02 * push_dir

        left_target = np.array([target_left_2d[0], target_left_2d[1], self.push_z])
        right_target = np.array([target_right_2d[0], target_right_2d[1], self.push_z])
        
        return left_target, right_target


# ==============================================================================
# 3. 数据采集与存储管道 (Diffusion Policy Zarr Standard)
# ==============================================================================
class DataCollector:
    def __init__(self, xml_path, zarr_save_path="push_t_demo.zarr"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 自动获取末端（请根据实际 openarm 骨骼改名，通常为 link7/hand/gripper）
        self.left_ee_name = "openarm_left_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7") != -1 else "left_hand"
        self.right_ee_name = "openarm_right_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7") != -1 else "right_hand"
        
        self.ik_solver = BimanualIKSolver(self.model, self.data, self.left_ee_name, self.right_ee_name)
        self.expert = BimanualPushTExpert()
        
        # 视觉渲染器 (64x64/128x128 符合 Diffusion 常见规格)
        self.camera_name = "top_cam"
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.zarr_path = zarr_save_path

    def reset_environment(self):
        """随机初始化 T 块和 Target，严格保证处于双臂公共操作区域"""
        mujoco.mj_resetData(self.model, self.data)
        
        # 1. 采样 T 块
        t_x = np.random.uniform(0.36, 0.46)
        t_y = np.random.uniform(-0.08, 0.08)
        t_yaw = np.random.uniform(-np.pi, np.pi)
        
        # 2. 采样 目标 Target
        g_x = np.random.uniform(0.38, 0.44)
        g_y = np.random.uniform(-0.06, 0.06)
        g_yaw = np.random.uniform(-np.pi/3, np.pi/3)

        # 写入 T 块状态到 Freejoint
        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()  # [x, y, z, w]
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.195]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]] # mujoco: [w, x, y, z]

        # 写入 目标虚影 (静止体)
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        quat_g = R.from_euler('z', g_yaw).as_quat()
        self.model.body_quat[goal_body_id] = [quat_g[3], quat_g[0], quat_g[1], quat_g[2]]

        mujoco.mj_forward(self.model, self.data)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw

    def collect_dataset(self, num_episodes=50, max_steps=200):
        """执行专家示教并以 Zarr 格式写入磁盘"""
        root = zarr.open(self.zarr_path, mode='w')
        data_grp = root.create_group('data')
        meta_grp = root.create_group('meta')

        obs_img_list = []
        obs_state_list = []
        action_list = []
        episode_ends = []
        total_steps = 0

        print(f"开始采集 {num_episodes} 条双臂 Push-T 轨迹...")

        for ep in range(num_episodes):
            t_pos, t_yaw, g_pos, g_yaw = self.reset_environment()
            
            for step in range(max_steps):
                # 1. 渲染图像观察
                self.renderer.update_scene(self.data, camera=self.camera_name)
                img = self.renderer.render()  # (128, 128, 3) uint8

                # 2. 抽取当前双机械手末端位置作为 Agent 低维状态
                l_pos = self.data.xpos[self.ik_solver.left_ee_id].copy()
                r_pos = self.data.xpos[self.ik_solver.right_ee_id].copy()
                agent_state = np.concatenate([l_pos, r_pos])  # (6,)

                # 3. 专家规划器生成 3D 目标点
                target_l, target_r = self.expert.get_action(t_pos, t_yaw, g_pos, g_yaw)
                
                # 动作定义：末端期望位姿目标 (6D: 左手 3D + 右手 3D)
                action = np.concatenate([target_l, target_r])

                # 4. IK 解算并驱动仿真
                target_q = self.ik_solver.solve(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q
                
                # 物理推进一步
                for _ in range(5):
                    mujoco.mj_step(self.model, self.data)

                # 5. 更新 T 块真实状态
                t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
                t_pos = self.data.qpos[t_joint_id:t_joint_id+3]
                q_wxyz = self.data.qpos[t_joint_id+3:t_joint_id+7]
                r_mat = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                t_yaw = r_mat.as_euler('zyx')[0]

                # 记录数据
                obs_img_list.append(img)
                obs_state_list.append(agent_state)
                action_list.append(action)
                total_steps += 1

                # 判断任务成功（距离 < 1.5cm 且 角度差 < 10 度）
                if np.linalg.norm(g_pos[:2] - t_pos[:2]) < 0.015 and abs((g_yaw - t_yaw + np.pi) % (2*np.pi) - np.pi) < np.deg2rad(10):
                    print(f"Episode {ep+1} 提前收敛，耗时 {step+1} 步！")
                    break

            episode_ends.append(total_steps)
            print(f"完成进度: [{ep+1}/{num_episodes}]")

        # 写入 Zarr 数据集 (Diffusion Policy 严格对应 Schema)
        data_grp.create_dataset('img', data=np.array(obs_img_list, dtype=np.uint8), chunks=(100, 128, 128, 3))
        data_grp.create_dataset('state', data=np.array(obs_state_list, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('action', data=np.array(action_list, dtype=np.float32), chunks=(1000, 6))
        meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

        print(f"\n全部采集完毕！数据已保存至：{os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    collector = DataCollector(xml_path="openarm_push_t.xml")
    collector.collect_dataset(num_episodes=10, max_steps=180)
