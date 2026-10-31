import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ==============================================================================
# 1. 迁移后的零空间投影 IK 动作执行器 (适配 24 维总物理状态)
# ==============================================================================
class BimanualOperationalSpaceIK:
    def __init__(self, model, data, left_ee_name, right_ee_name):
        self.model = model
        self.data = data
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)
        self.use_site = False
        
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_ee_name)
            self.use_site = True

        # OpenArm 经典工作姿态（微屈臂，肘部外扩，防止奇异点）
        self.q_nominal = np.zeros(self.model.nu)
        if self.model.nu >= 14:
            self.q_nominal[:7] = [0.0, 0.35, 0.0, -1.1, 0.0, 0.45, 0.0]
            self.q_nominal[7:14] = [0.0, -0.35, 0.0, -1.1, 0.0, 0.45, 0.0]

    def solve(self, left_target, right_target, max_steps=3, damping=0.005):
        for _ in range(max_steps):
            curr_l = self.data.site_xpos[self.left_ee_id] if self.use_site else self.data.xpos[self.left_ee_id]
            curr_r = self.data.site_xpos[self.right_ee_id] if self.use_site else self.data.xpos[self.right_ee_id]

            err_l = left_target - curr_l
            err_r = right_target - curr_r
            error = np.clip(np.concatenate([err_l, err_r]), -0.04, 0.04)

            if np.linalg.norm(error) < 0.002:
                break

            jacp_l, jacp_r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if self.use_site:
                mujoco.mj_jacSite(self.model, self.data, jacp_l, jacr, self.left_ee_id)
                mujoco.mj_jacSite(self.model, self.data, jacp_r, jacr, self.right_ee_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr, self.left_ee_id)
                mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr, self.right_ee_id)

            J = np.vstack([jacp_l, jacp_r])  # (6, 24)
            reg = damping * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg)  # (24, 6)

            dq_main = J_pinv @ error

            # 零空间自构形姿态引力
            I = np.eye(self.model.nv)
            dq_null = np.zeros(self.model.nv)
            dq_null[:self.model.nu] = 0.4 * (self.q_nominal - self.data.qpos[:self.model.nu])

            dq = dq_main + (I - J_pinv @ J) @ dq_null
            dq[self.model.nu:] = 0.0  # 锁定物体自由度
            dq = np.clip(dq, -0.04, 0.04)

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)

        return self.data.qpos[:self.model.nu].copy()


# ==============================================================================
# 2. 接触导向的样条动作示教模型 (Contact-Aware Spline Motion Model)
# ==============================================================================
class BimanualPushTrajectoryPlanner:
    def __init__(self, push_z=0.194, hover_z=0.235):
        self.push_z = push_z
        self.hover_z = hover_z

    def generate_trajectory(self, t_init_pos, t_init_yaw, goal_pos, goal_yaw):
        """一次性生成整条从起点推至目标区的平滑 3D 轨迹点序列"""
        waypoints_l, waypoints_r = [], []
        
        diff_yaw = (goal_yaw - t_init_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pos = goal_pos[:2] - t_init_pos[:2]
        dist = np.linalg.norm(diff_pos)

        # -------------------------------------------------------------
        # 阶段 A: 调姿态（如果初始角度偏差大于 12 度）
        # -------------------------------------------------------------
        if abs(diff_yaw) > np.deg2rad(12):
            t_dir = np.array([np.cos(t_init_yaw), np.sin(t_init_yaw)])
            t_perp = np.array([-np.sin(t_init_yaw), np.cos(t_init_yaw)])
            
            p_l_contact = t_init_pos[:2] + 0.07 * t_dir
            p_r_contact = t_init_pos[:2] - 0.07 * t_dir
            
            # 1. 悬停就位点
            p_l_hover = np.array([p_l_contact[0], p_l_contact[1], self.hover_z])
            p_r_hover = np.array([p_r_contact[0], p_r_contact[1], self.hover_z])
            # 2. 下降接触点
            p_l_touch = np.array([p_l_contact[0], p_l_contact[1], self.push_z])
            p_r_touch = np.array([p_r_contact[0], p_r_contact[1], self.push_z])
            # 3. 错动旋转点
            sign = 1.0 if diff_yaw > 0 else -1.0
            p_l_turn = p_l_touch + np.array([-sign * 0.03 * t_perp[0], -sign * 0.03 * t_perp[1], 0.0])
            p_r_turn = p_r_touch + np.array([ sign * 0.03 * t_perp[0],  sign * 0.03 * t_perp[1], 0.0])

            # 生成阶段 A 样条插值点
            self._interpolate(waypoints_l, waypoints_r, p_l_hover, p_r_hover, steps=25)
            self._interpolate(waypoints_l, waypoints_r, p_l_touch, p_r_touch, steps=15)
            self._interpolate(waypoints_l, waypoints_r, p_l_turn, p_r_turn, steps=30)
            self._interpolate(waypoints_l, waypoints_r, p_l_hover, p_r_hover, steps=15)  # 抬手

        # -------------------------------------------------------------
        # 阶段 B: 直线平移推进 (双臂一左一右对称匀速向前推入目标区)
        # -------------------------------------------------------------
        push_dir = diff_pos / (dist + 1e-6)
        push_perp = np.array([-push_dir[1], push_dir[0]])
        
        # 站在物体背后 5.5cm 处
        behind_start = t_init_pos[:2] - 0.055 * push_dir
        behind_end = goal_pos[:2] - 0.02 * push_dir

        # 起始接触点
        p_l_start = np.array([behind_start[0] + 0.04 * push_perp[0], behind_start[1] + 0.04 * push_perp[1], self.push_z])
        p_r_start = np.array([behind_start[0] - 0.04 * push_perp[0], behind_start[1] - 0.04 * push_perp[1], self.push_z])
        
        # 终点接触点
        p_l_end = np.array([behind_end[0] + 0.04 * push_perp[0], behind_end[1] + 0.04 * push_perp[1], self.push_z])
        p_r_end = np.array([behind_end[0] - 0.04 * push_perp[0], behind_end[1] - 0.04 * push_perp[1], self.push_z])

        # 悬停点
        p_l_hover = np.array([p_l_start[0], p_l_start[1], self.hover_z])
        p_r_hover = np.array([p_r_start[0], p_r_start[1], self.hover_z])

        # 生成阶段 B 样条插值点
        self._interpolate(waypoints_l, waypoints_r, p_l_hover, p_r_hover, steps=25)
        self._interpolate(waypoints_l, waypoints_r, p_l_start, p_r_start, steps=15)
        # 根据推行距离动态调整推行步数，保证匀速推进（约 2mm/步）
        push_steps = max(int(dist / 0.002), 40)
        self._interpolate(waypoints_l, waypoints_r, p_l_end, p_r_end, steps=push_steps)
        # 抬起脱离
        p_l_final_hover = np.array([p_l_end[0], p_l_end[1], self.hover_z])
        p_r_final_hover = np.array([p_r_end[0], p_r_end[1], self.hover_z])
        self._interpolate(waypoints_l, waypoints_r, p_l_final_hover, p_r_final_hover, steps=15)

        return waypoints_l, waypoints_r

    def _interpolate(self, list_l, list_r, target_l, target_r, steps=20):
        """采用最小加加速度余弦平滑插值 (S 曲线)，保证速度连续无冲击"""
        start_l = list_l[-1] if len(list_l) > 0 else target_l
        start_r = list_r[-1] if len(list_r) > 0 else target_r
        for i in range(1, steps + 1):
            t = i / float(steps)
            s = 0.5 * (1.0 - np.cos(np.pi * t))  # 余弦加速减速过渡
            interp_l = start_l + s * (target_l - start_l)
            interp_r = start_r + s * (target_r - start_r)
            list_l.append(interp_l)
            list_r.append(interp_r)


# ==============================================================================
# 3. 数据采集与验证系统
# ==============================================================================
class BimanualDataCollector:
    def __init__(self, xml_path="openarm_pusht.xml", zarr_save_path="push_t_demo.zarr"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.left_ee_name = "openarm_left_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7") != -1 else "left_hand"
        self.right_ee_name = "openarm_right_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7") != -1 else "right_hand"

        self.ik_solver = BimanualOperationalSpaceIK(self.model, self.data, self.left_ee_name, self.right_ee_name)
        self.planner = BimanualPushTrajectoryPlanner()

        self.camera_name = "top_cam"
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.zarr_path = zarr_save_path

    def check_success(self, t_pos, t_yaw, goal_pos, goal_yaw):
        """严格成功判据：终点位置误差 < 1.8cm 且 角度差 < 10°"""
        pos_err = np.linalg.norm(goal_pos[:2] - t_pos[:2])
        yaw_err = abs((goal_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi)
        return (pos_err < 0.018 and yaw_err < np.deg2rad(10.0)), pos_err, yaw_err

    def reset_environment(self):
        mujoco.mj_resetData(self.model, self.data)

        # 随机化 T 块与目标位姿（确保在双臂安全交叠区内）
        t_x = np.random.uniform(0.38, 0.44)
        t_y = np.random.uniform(-0.05, 0.05)
        t_yaw = np.random.uniform(-np.pi/3, np.pi/3)

        g_x = np.random.uniform(0.39, 0.43)
        g_y = np.random.uniform(-0.03, 0.03)
        g_yaw = np.random.uniform(-np.pi/4, np.pi/4)

        t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
        quat_t = R.from_euler('z', t_yaw).as_quat()
        t_quat_wxyz = [quat_t[3], quat_t[0], quat_t[1], quat_t[2]]
        self.data.qpos[t_joint_id:t_joint_id+3] = [t_x, t_y, 0.195]
        self.data.qpos[t_joint_id+3:t_joint_id+7] = t_quat_wxyz

        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
        quat_g = R.from_euler('z', g_yaw).as_quat()
        g_quat_wxyz = [quat_g[3], quat_g[0], quat_g[1], quat_g[2]]
        self.model.body_pos[goal_body_id] = [g_x, g_y, 0.181]
        self.model.body_quat[goal_body_id] = g_quat_wxyz

        # 手臂复位到中立名义姿态
        self.data.qpos[:len(self.ik_solver.q_nominal)] = self.ik_solver.q_nominal
        mujoco.mj_forward(self.model, self.data)

        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181] + g_quat_wxyz, dtype=np.float32)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect_dataset(self, num_episodes=10):
        obs_img_list, obs_state_list, action_list = [], [], []
        t_init_list, goal_init_list, episode_ends = [], [], []

        successful_episodes = 0
        total_attempts = 0
        total_steps = 0
        debug_gif_frames = []

        print(f"\n🚀 启动双臂样条动作示教模型采集（目标: {num_episodes} 条成功轨迹）...")

        while successful_episodes < num_episodes:
            total_attempts += 1
            t_pos, t_yaw, g_pos, g_yaw, t_init_7d, goal_init_7d = self.reset_environment()

            # 由动作模型一次性规划出平滑全局轨迹
            waypoints_l, waypoints_r = self.planner.generate_trajectory(t_pos, t_yaw, g_pos, g_yaw)

            ep_imgs, ep_states, ep_actions = [], [], []

            for step_i in range(len(waypoints_l)):
                self.renderer.update_scene(self.data, camera=self.camera_name)
                img = self.renderer.render()

                l_pos = self.data.xpos[self.ik_solver.left_ee_id].copy()
                r_pos = self.data.xpos[self.ik_solver.right_ee_id].copy()
                agent_state = np.concatenate([l_pos, r_pos])

                target_l = waypoints_l[step_i]
                target_r = waypoints_r[step_i]
                action = np.concatenate([target_l, target_r])

                target_q = self.ik_solver.solve(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

                ep_imgs.append(img)
                ep_states.append(agent_state)
                ep_actions.append(action)

            # 获取物理仿真推进后的最终 T 块位姿
            t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
            final_t_pos = self.data.qpos[t_joint_id:t_joint_id+3]
            q_wxyz = self.data.qpos[t_joint_id+3:t_joint_id+7]
            r_mat = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            final_t_yaw = r_mat.as_euler('zyx')[0]

            success, p_err, y_err = self.check_success(final_t_pos, final_t_yaw, g_pos, g_yaw)

            if success:
                obs_img_list.extend(ep_imgs)
                obs_state_list.extend(ep_states)
                action_list.extend(ep_actions)
                t_init_list.append(t_init_7d)
                goal_init_list.append(goal_init_7d)
                total_steps += len(ep_actions)
                episode_ends.append(total_steps)

                if successful_episodes == 0 and HAS_IMAGEIO:
                    debug_gif_frames = ep_imgs.copy()

                successful_episodes += 1
                print(f"✔ 轨迹 #{successful_episodes} 录制达标！用时 {len(ep_actions)} 步 | 距离误差: {p_err*100:.1f}cm, 角度误差: {np.rad2deg(y_err):.1f}°")
            else:
                print(f"✘ 尝试 #{total_attempts}: 最终误差稍大 (距离 {p_err*100:.1f}cm)，自动剔除重新规划...")

        # 写入 Zarr 数据集 (Diffusion Policy 规范)
        root = zarr.open(self.zarr_path, mode='w')
        data_grp = root.create_group('data')
        meta_grp = root.create_group('meta')
        data_grp.create_dataset('img', data=np.array(obs_img_list, dtype=np.uint8), chunks=(100, 128, 128, 3))
        data_grp.create_dataset('state', data=np.array(obs_state_list, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('action', data=np.array(action_list, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('t_init_pose', data=np.array(t_init_list, dtype=np.float32))
        data_grp.create_dataset('goal_pose', data=np.array(goal_init_list, dtype=np.float32))
        meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

        if HAS_IMAGEIO and len(debug_gif_frames) > 0:
            gif_path = "debug_episode.gif"
            imageio.mimsave(gif_path, debug_gif_frames[::2], fps=20)
            print(f"已导出首条高质量轨迹动图: {os.path.abspath(gif_path)}")

        print(f"\n全部 {num_episodes} 条高质量示教轨迹采集完成！已存至: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")

    collector = BimanualDataCollector(xml_path=xml_path)
    collector.collect_dataset(num_episodes=10)
