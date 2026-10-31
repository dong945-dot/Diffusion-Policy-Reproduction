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
# 1. 刚性手腕锁定的双臂闭环 IK 执行器 (确保夹爪垂直朝下，绝不翻滚侧倾)
# ==============================================================================
class BimanualRigidWristIK:
    def __init__(self, model, data, left_ee_name, right_ee_name):
        self.model = model
        self.data = data
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)
        self.use_site = False
        
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_ee_name)
            self.use_site = True

        # 预设标准垂直端正工作姿态
        self.q_nominal = np.zeros(self.model.nu)
        if self.model.nu >= 14:
            # 严格锁定肘关节外倾与手腕垂直下指姿态
            self.q_nominal[:7] = [0.0, 0.45, 0.0, -1.25, 0.0, 0.8, 0.0]
            self.q_nominal[7:14] = [0.0, -0.45, 0.0, -1.25, 0.0, 0.8, 0.0]

    def solve(self, left_target, right_target, max_steps=4, damping=0.005):
        for _ in range(max_steps):
            curr_l = self.data.site_xpos[self.left_ee_id] if self.use_site else self.data.xpos[self.left_ee_id]
            curr_r = self.data.site_xpos[self.right_ee_id] if self.use_site else self.data.xpos[self.right_ee_id]

            err_l = left_target - curr_l
            err_r = right_target - curr_r
            error = np.clip(np.concatenate([err_l, err_r]), -0.03, 0.03)

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
            J_pinv = J.T @ np.linalg.inv(J @ J.T + reg)

            dq_main = J_pinv @ error

            # 强手腕约束：对后部关节（手腕）施加极大归位权重，保证手爪严格垂直朝下
            I = np.eye(self.model.nv)
            dq_null = np.zeros(self.model.nv)
            weights = np.ones(self.model.nu) * 0.3
            if self.model.nu >= 14:
                # 强化手腕关节阻尼权重
                weights[4:7] = 0.8
                weights[11:14] = 0.8
            dq_null[:self.model.nu] = weights * (self.q_nominal - self.data.qpos[:self.model.nu])

            dq = dq_main + (I - J_pinv @ J) @ dq_null
            dq[self.model.nu:] = 0.0  # 严防扰动 T 块自身自由度
            dq = np.clip(dq, -0.035, 0.035)

            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)
            mujoco.mj_kinematics(self.model, self.data)

        return self.data.qpos[:self.model.nu].copy()


# ==============================================================================
# 2. 闭环反应式推 T 专家策略 (Closed-Loop Reactive Expert)
# ==============================================================================
class ClosedLoopPushExpert:
    def __init__(self, push_z=0.193, hover_z=0.235):
        self.push_z = push_z
        self.hover_z = hover_z
        self.curr_l = None
        self.curr_r = None

    def reset(self, initial_l, initial_r):
        self.curr_l = np.array(initial_l, dtype=np.float64)
        self.curr_r = np.array(initial_r, dtype=np.float64)

    def step(self, t_curr_pos, t_curr_yaw, goal_pos, goal_yaw):
        """每一帧都根据 T 块的【实时真实位置】动态调整落点与推力"""
        diff_yaw = (goal_yaw - t_curr_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pos = goal_pos[:2] - t_curr_pos[:2]
        dist = np.linalg.norm(diff_pos)

        # -------------------------------------------------------------
        # 模式 1: 姿态偏差过大 -> 启动原地闭环拧转
        # -------------------------------------------------------------
        if abs(diff_yaw) > np.deg2rad(10.0):
            t_dir = np.array([np.cos(t_curr_yaw), np.sin(t_curr_yaw)])
            t_perp = np.array([-np.sin(t_curr_yaw), np.cos(t_curr_yaw)])
            
            # 紧贴 T 块横梁左右两侧
            pt_left = t_curr_pos[:2] + 0.065 * t_dir
            pt_right = t_curr_pos[:2] - 0.065 * t_dir
            
            sign = 1.0 if diff_yaw > 0 else -1.0
            # 形成反向力偶
            tgt_l_2d = pt_left - sign * 0.035 * t_perp
            tgt_r_2d = pt_right + sign * 0.035 * t_perp

            desired_l = np.array([tgt_l_2d[0], tgt_l_2d[1], self.push_z])
            desired_r = np.array([tgt_r_2d[0], tgt_r_2d[1], self.push_z])
            phase = "ROTATE"

        # -------------------------------------------------------------
        # 模式 2: 姿态已对准 -> 动态咬住 T 块后方协同向前推
        # -------------------------------------------------------------
        else:
            push_dir = diff_pos / (dist + 1e-6)
            push_perp = np.array([-push_dir[1], push_dir[0]])
            
            # 严格位于【当前 T 块】后方 4.5cm 处
            contact_center = t_curr_pos[:2] - 0.045 * push_dir
            
            # 双手分开 7cm 形成闭环“卡槽”
            tgt_l_2d = contact_center + 0.035 * push_perp + 0.03 * push_dir
            tgt_r_2d = contact_center - 0.035 * push_perp + 0.03 * push_dir

            desired_l = np.array([tgt_l_2d[0], tgt_l_2d[1], self.push_z])
            desired_r = np.array([tgt_r_2d[0], tgt_r_2d[1], self.push_z])
            phase = "PUSH"

        # -------------------------------------------------------------
        # 闭环平滑阻尼器（限制单步最大位移 2.5mm，保持丝滑）
        # -------------------------------------------------------------
        max_step_dist = 0.0025
        delta_l = desired_l - self.curr_l
        dist_l = np.linalg.norm(delta_l)
        if dist_l > max_step_dist:
            delta_l = delta_l / dist_l * max_step_dist

        delta_r = desired_r - self.curr_r
        dist_r = np.linalg.norm(delta_r)
        if dist_r > max_step_dist:
            delta_r = delta_r / dist_r * max_step_dist

        self.curr_l += delta_l
        self.curr_r += delta_r

        return self.curr_l.copy(), self.curr_r.copy(), phase


# ==============================================================================
# 3. 闭环采集与成功筛选系统
# ==============================================================================
class BimanualDataCollector:
    def __init__(self, xml_path="openarm_pusht.xml", zarr_save_path="push_t_demo.zarr"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.left_ee_name = "openarm_left_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7") != -1 else "left_hand"
        self.right_ee_name = "openarm_right_link7" if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7") != -1 else "right_hand"

        self.ik_solver = BimanualRigidWristIK(self.model, self.data, self.left_ee_name, self.right_ee_name)
        self.expert = ClosedLoopPushExpert()

        self.camera_name = "top_cam"
        self.renderer = mujoco.Renderer(self.model, height=128, width=128)
        self.zarr_path = zarr_save_path

    def check_success(self, t_pos, t_yaw, goal_pos, goal_yaw):
        """严格达标判据：距离误差 < 1.6cm 且 角度差 < 8.0°"""
        pos_err = np.linalg.norm(goal_pos[:2] - t_pos[:2])
        yaw_err = abs((goal_yaw - t_yaw + np.pi) % (2 * np.pi) - np.pi)
        return (pos_err < 0.016 and yaw_err < np.deg2rad(8.0)), pos_err, yaw_err

    def reset_environment(self):
        mujoco.mj_resetData(self.model, self.data)

        # 随机化 T 块位姿
        t_x = np.random.uniform(0.38, 0.43)
        t_y = np.random.uniform(-0.04, 0.04)
        t_yaw = np.random.uniform(-np.pi/3, np.pi/3)

        # 随机化 目标位姿
        g_x = np.random.uniform(0.39, 0.42)
        g_y = np.random.uniform(-0.02, 0.02)
        g_yaw = np.random.uniform(-np.pi/5, np.pi/5)

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

        # 复位机器人姿态
        self.data.qpos[:len(self.ik_solver.q_nominal)] = self.ik_solver.q_nominal
        mujoco.mj_forward(self.model, self.data)

        # 初始化闭环规划器的起点
        l_init = self.data.xpos[self.ik_solver.left_ee_id].copy()
        r_init = self.data.xpos[self.ik_solver.right_ee_id].copy()
        self.expert.reset(l_init, r_init)

        t_init_7d = np.array([t_x, t_y, 0.195] + t_quat_wxyz, dtype=np.float32)
        goal_init_7d = np.array([g_x, g_y, 0.181] + g_quat_wxyz, dtype=np.float32)
        return np.array([t_x, t_y, 0.195]), t_yaw, np.array([g_x, g_y, 0.181]), g_yaw, t_init_7d, goal_init_7d

    def collect_dataset(self, num_episodes=10, max_steps=260):
        obs_img_list, obs_state_list, action_list = [], [], []
        t_init_list, goal_init_list, episode_ends = [], [], []

        successful_episodes = 0
        total_attempts = 0
        total_steps = 0
        debug_gif_frames = []

        print(f"\n🚀 启动【闭环反应式】双臂推 T 示教采集（目标: {num_episodes} 条高质量轨迹）...")

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

                # 💡 核心：每一步都向专家输入【此时此刻真实 T 块的实时坐标】，杜绝盲推
                target_l, target_r, phase = self.expert.step(t_pos, t_yaw, g_pos, g_yaw)
                action = np.concatenate([target_l, target_r])

                target_q = self.ik_solver.solve(target_l, target_r)
                self.data.ctrl[:len(target_q)] = target_q

                for _ in range(4):
                    mujoco.mj_step(self.model, self.data)

                # 实时刷新读取 T 块最新真实位姿
                t_joint_id = self.model.joint("tee_freejoint").qposadr[0]
                t_pos = self.data.qpos[t_joint_id:t_joint_id+3]
                q_wxyz = self.data.qpos[t_joint_id+3:t_joint_id+7]
                r_mat = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                t_yaw = r_mat.as_euler('zyx')[0]

                ep_imgs.append(img)
                ep_states.append(agent_state)
                ep_actions.append(action)

                # 实时判定是否达成目标
                success, p_err, y_err = self.check_success(t_pos, t_yaw, g_pos, g_yaw)
                if success:
                    ep_success = True
                    print(f"✔ 尝试 #{total_attempts}: 闭环推正成功！耗时 {step+1} 步 | 距离误差: {p_err*100:.1f}cm, 角度误差: {np.rad2deg(y_err):.1f}°")
                    break

            if ep_success:
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
                print(f"进度: [{successful_episodes}/{num_episodes}] (当前有效成功率: {successful_episodes/total_attempts*100:.1f}%)")
            else:
                print(f"✘ 尝试 #{total_attempts}: 超时脱靶，自动丢弃重试...")

        # 写入 Zarr 数据集
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
            print(f"已导出首条高质量闭环示教动图: {os.path.abspath(gif_path)}")

        print(f"\n全部 {num_episodes} 条高质量示教轨迹采集完成！已存入: {os.path.abspath(self.zarr_path)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")

    collector = BimanualDataCollector(xml_path=xml_path)
    collector.collect_dataset(num_episodes=10, max_steps=260)
