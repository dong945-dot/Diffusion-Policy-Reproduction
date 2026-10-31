import os
import zarr
import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# 尝试导入 imageio 用于自动录制 debug.gif
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ==============================================================================
# 1. 带有“零空间姿态保护”的高级 IK 求解器 (杜绝关节翻滚抽搐)
# ==============================================================================
class SmoothBimanualIKSolver:
    def __init__(self, model, data, left_ee_name, right_ee_name):
        self.model = model
        self.data = data
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)
        
        self.use_site = False
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_ee_name)
            self.use_site = True

        # 默认中立舒适姿态（名义姿态：双臂自然微曲悬停，杜绝肘部奇形怪状）
        self.q_nominal = np.zeros(self.model.nu)
        if self.model.nu >= 14:
            # 预设一个典型的人形双臂自然下垂姿态
            self.q_nominal[:7] = [0.0, 0.4, 0.0, -1.2, 0.0, 0.5, 0.0]
            self.q_nominal[7:14] = [0.0, -0.4, 0.0, -1.2, 0.0, 0.5, 0.0]

    def solve(self, left_target_pos, right_target_pos, max_steps=4, damping=0.005):
        for _ in range(max_steps):
            curr_l = self.data.site_xpos[self.left_ee_id] if self.use_site else self.data.xpos[self.left_ee_id]
            curr_r = self.data.site_xpos[self.right_ee_id] if self.use_site else self.data.xpos[self.right_ee_id]

            # 计算末端误差并限幅
            err_l = left_target_pos - curr_l
            err_r = right_target_pos - curr_r
            error = np.clip(np.concatenate([err_l, err_r]), -0.05, 0.05)

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

            J = np.vstack([jacp_l, jacp_r])  # (6, nv)
            
            # 阻尼最小二乘伪逆: J_pinv = J^T (J J^T + lambda^2 I)^-1
            reg = damping * np.eye(6)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg)

            # 主任务增量
            dq_main = J_pinv @ error

            # 零空间自构形控制 (Null-space projection): 把手臂往好看自然的中立姿态拉
            I = np.eye(self.model.nv)
            q_current = self.data.qpos[:self.model.nu]
            dq_null = 0.5 * (self.q_nominal - q_current)
            
            # 合并主任务与零空间任务
            dq = dq_main + (I - J_pinv @ J) @ dq_null
            dq = np.clip(dq, -0.05, 0.05)  # 严格限制单步关节跳变

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)

        return self.data.qpos[:self.model.nu].copy()


# ==============================================================================
# 2. 状态机平滑推 T 专家策略 (带高度抬升与平滑滤波)
# ==============================================================================
class SmoothPushTExpert:
    def __init__(self, push_height=0.194, hover_height=0.24):
        self.push_z = push_height
        self.hover_z = hover_height
        self.curr_l_target = None
        self.curr_r_target = None

    def plan_targets(self, t_pos, t_yaw, goal_pos, goal_yaw):
        diff_yaw = (goal_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pos = goal_pos[:2] - t_pos[:2]
        dist = np.linalg.norm(diff_pos)

        # -----------------------------------------------------------------
        # 阶段 1: 姿态微调对齐 (两手在 T 侧面轻柔点推纠正旋转)
        # -----------------------------------------------------------------
        if abs(diff_yaw) > np.deg2rad(10):
            t_dir = np.array([np.cos(t_yaw), np.sin(t_yaw)])
            t_perp = np.array([-np.sin(t_yaw), np.cos(t_yaw)])
            
            # 位于横梁两侧
            pt_left = t_pos[:2] + 0.07 * t_dir
            pt_right = t_pos[:2] - 0.07 * t_dir
            
            if diff_yaw > 0:  # 逆时针修正
                target_l = np.array([pt_left[0] - 0.025 * t_perp[0], pt_left[1] - 0.025 * t_perp[1], self.push_z])
                target_r = np.array([pt_right[0] + 0.025 * t_perp[0], pt_right[1] + 0.025 * t_perp[1], self.push_z])
            else:             # 顺时针修正
                target_l = np.array([pt_left[0] + 0.025 * t_perp[0], pt_left[1] + 0.025 * t_perp[1], self.push_z])
                target_r = np.array([pt_right[0] - 0.025 * t_perp[0], pt_right[1] - 0.025 * t_perp[1], self.push_z])
            phase = "ROTATION"
        
        # -----------------------------------------------------------------
        # 阶段 2: 直线平移推进 (双臂如拱门般一左一右对称匀速向前推)
        # -----------------------------------------------------------------
        else:
            push_dir = diff_pos / (dist + 1e-6)
            push_perp = np.array([-push_dir[1], push_dir[0]])
            
            # 紧贴 T 块底后方，左右手间距 8cm
            base_pt = t_pos[:2] - 0.05 * push_dir
            pt_l = base_pt + 0.04 * push_perp
            pt_r = base_pt - 0.04 * push_perp
            
            target_l = np.array([pt_l[0], pt_l[1], self.push_z])
            target_r = np.array([pt_r[0], pt_r[1], self.push_z])
            phase = "TRANSLATION"

        # -----------------------------------------------------------------
        # 动作连续性平滑器 (EMA 滤波器，消除跳变)
        # -----------------------------------------------------------------
        if self.curr_l_target is None:
            self.curr_l_target = target_l
            self.curr_r_target = target_r
        else:
            # 限制单步最大位移不超过 4mm，杜绝瞬移和抽搐
            delta_l = np.clip(target_l - self.curr_l_target, -0.005, 0.005)
            delta_r = np.clip(target_r - self.curr_r_target, -0.005, 0.005)
            self.curr_l_target += delta_l
            self.curr_r_target += delta_r

        return self.curr_l_target.copy(), self.curr_r_target.copy(), phase


# ==============================================================================
# 3. 严格成功标准 + 数据采集器
# ==============================================================================
class DataCollector:
    def __init__(self, xml_path="openarm_pusht.xml", zarr_save_path="push_t_demo.zarr"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.left_ee_name = "openarm_left_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7") != -1 else "left_hand"
        self.right_ee_name = "openarm_right_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7") != -1 else "right_hand"
        
        self.ik_solver = SmoothBimanualIKSolver(self.model, self.data, self.left_ee_name, self.right_ee_name)
        self.expert = SmoothPushTExpert()
        
        self.camera_name = "top_cam"
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.zarr_path = zarr_save_path

    def check_success(self, t_pos, t_yaw, goal_pos, goal_yaw):
        """严格成功判据：位置误差 < 1.5 cm 且 角度误差 < 8.5 度"""
        pos_err = np.linalg.norm(goal_pos[:2] - t_pos[:2])
        yaw_err = abs((goal_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi)
        
        is_pos_success = pos_err < 0.015
        is_yaw_success = yaw_err < np.deg2rad(8.5)
        return is_pos_success and is_yaw_success, pos_err, yaw_err

    def reset_environment(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机采样安全区域内的位置
        t_x = np.random.uniform(0.38, 0.44)
        t_y = np.random.uniform(-0.06, 0.06)
        t_yaw = np.random.uniform(-np.pi/2, np.pi/2)
        
        g_x = np.random.uniform(0.39, 0.43)
        g_y = np.random.uniform(-0.04, 0.04)
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

        # 默认手臂复位到名义姿态
        self.data.qpos[:len(self.ik_solver.q_nominal)] = self.ik_solver.q_nominal
        mujoco.mj_forward(self.model, self.data)

        # 重置专家的平滑滤波器
        self.expert.curr_l_target = None
        self.expert.curr_r_target = None

        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181] + g_quat_wxyz, dtype=np.float32)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect_dataset(self, num_episodes=10, max_steps=220):
        obs_img_list, obs_state_list, action_list = [], [], []
        t_init_list, goal_init_list, episode_ends = [], [], []
        
        successful_episodes = 0
        total_attempts = 0
        total_steps = 0
        debug_gif_frames = []

        print(f"\n开始收集高质量示教数据（目标: {num_episodes} 条有效成功轨迹）...")

        while successful_episodes < num_episodes:
            total_attempts += 1
            t_pos, t_yaw, g_pos, g_yaw, t_init_7d, goal_init_7d = self.reset_environment()

            ep_imgs, ep_states, ep_actions = [], [], []
            ep_success = False

            for step in range(max_steps):
                self.renderer.update_scene(self.data, camera=self.camera_name)
                img = self.renderer.render()

                l_pos = self.data.xpos[self.ik_solver.left_ee_id].copy()
                r_pos = self.data.xpos[self.ik_solver.right_ee_id].copy()
                agent_state = np.concatenate([l_pos, r_pos])

                # 专家规划（平滑插值）
                target_l, target_r, phase = self.expert.plan_targets(t_pos, t_yaw, g_pos, g_yaw)
                action = np.concatenate([target_l, target_r])

                target_q = self.ik_solver.solve(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

                # 更新 T 块位姿
                t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
                t_pos = self.data.qpos[t_joint_id:t_joint_id+3]
                q_wxyz = self.data.qpos[t_joint_id+3:t_joint_id+7]
                r_mat = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                t_yaw = r_mat.as_euler('zyx')[0]

                ep_imgs.append(img)
                ep_states.append(agent_state)
                ep_actions.append(action)

                # 判定成功标准
                success, p_err, y_err = self.check_success(t_pos, t_yaw, g_pos, g_yaw)
                if success:
                    ep_success = True
                    print(f"✔ 尝试 #{total_attempts}: 成功达成标准！耗时 {step+1} 步 | 终点误差: 距离 {p_err*100:.1f}cm, 角度 {np.rad2deg(y_err):.1f}°")
                    break

            # 录制第一条轨迹作为可视化 GIF 供检查
            if successful_episodes == 0 and HAS_IMAGEIO:
                debug_gif_frames = ep_imgs.copy()

            # 只有达到成功标准的轨迹才会被存入数据集
            if ep_success:
                obs_img_list.extend(ep_imgs)
                obs_state_list.extend(ep_states)
                action_list.extend(ep_actions)
                t_init_list.append(t_init_7d)
                goal_init_list.append(goal_init_7d)
                total_steps += len(ep_actions)
                episode_ends.append(total_steps)
                successful_episodes += 1
                print(f"进度: [{successful_episodes}/{num_episodes}] (成功率: {successful_episodes/total_attempts*100:.1f}%)")
            else:
                print(f"✘ 尝试 #{total_attempts}: 超时未达标，丢弃脏数据重试...")

        # 保存 Zarr 数据集
        root = zarr.open(self.zarr_path, mode='w')
        data_grp = root.create_group('data')
        meta_grp = root.create_group('meta')
        data_grp.create_dataset('img', data=np.array(obs_img_list, dtype=np.uint8), chunks=(100, 128, 128, 3))
        data_grp.create_dataset('state', data=np.array(obs_state_list, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('action', data=np.array(action_list, dtype=np.float32), chunks=(1000, 6))
        data_grp.create_dataset('t_init_pose', data=np.array(t_init_list, dtype=np.float32))
        data_grp.create_dataset('goal_pose', data=np.array(goal_init_list, dtype=np.float32))
        meta_grp.create_dataset('episode_ends', data=np.array(episode_ends, dtype=np.int64))

        # 保存 debug.gif
        if HAS_IMAGEIO and len(debug_gif_frames) > 0:
            gif_path = "debug_episode.gif"
            imageio.mimsave(gif_path, debug_gif_frames[::2], fps=20)
            print(f"\n已自动生成首条成功轨迹动图: {os.path.abspath(gif_path)}")

        print(f"\n全部成功数据已采集入库！路径: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")
    
    collector = DataCollector(xml_path=xml_path)
    collector.collect_dataset(num_episodes=10, max_steps=220)
