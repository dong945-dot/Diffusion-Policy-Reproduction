import os
import time
import zarr
import numpy as np
import mujoco
import mujoco.viewer

# 尝试导入 OpenCV 用于同步画中画展示录制的第一视角图像
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ==============================================================================
# 1. 双臂逆运动学 (IK) 求解器
# ==============================================================================
class BimanualIKSolver:
    def __init__(self, model, data, left_ee_name, right_ee_name, damping=1e-3):
        self.model = model
        self.data = data
        self.damping = damping
        self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
        self.right_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)
        self.use_site = False
        
        # 兼容末端为 site 的情况
        if self.left_ee_id == -1:
            self.left_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, left_ee_name)
            self.use_site = True

    def solve(self, left_target_pos, right_target_pos, max_steps=4):
        for _ in range(max_steps):
            curr_l = self.data.site_xpos[self.left_ee_id] if self.use_site else self.data.xpos[self.left_ee_id]
            curr_r = self.data.site_xpos[self.right_ee_id] if self.use_site else self.data.xpos[self.right_ee_id]

            err_l = left_target_pos - curr_l
            err_r = right_target_pos - curr_r
            error = np.concatenate([err_l, err_r])

            if np.linalg.norm(error) < 0.003:
                break

            jacp_l, jacp_r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
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
# 2. 物理与视觉动态回放函数
# ==============================================================================
def replay_dataset(zarr_path="push_t_demo.zarr", xml_path="openarm_push_t.xml"):
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"未找到数据集文件: {zarr_path}，请先运行采集脚本！")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"未找到场景 XML 文件: {xml_path}")

    # 读取 Zarr 数据集
    print(f"正在加载数据: {zarr_path} ...")
    root = zarr.open(zarr_path, mode='r')
    actions = root['data/action'][:]
    episode_ends = root['meta/episode_ends'][:]
    
    # 检查是否保存了 T 块与目标的初始位姿
    has_t_init = 't_init_pose' in root['data']
    has_goal_init = 'goal_pose' in root['data']
    if has_t_init:
        t_init_poses = root['data/t_init_pose'][:]
    if has_goal_init:
        goal_poses = root['data/goal_pose'][:]
        
    has_imgs = 'img' in root['data']
    if has_imgs:
        imgs = root['data/img']

    num_episodes = len(episode_ends)
    print(f"成功载入！共包含 {num_episodes} 条示教轨迹，总步数: {actions.shape[0]}。")

    # 初始化 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    left_ee_name = "openarm_left_link7" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_left_link7") != -1 else "left_hand"
    right_ee_name = "openarm_right_link7" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_right_link7") != -1 else "right_hand"
    ik_solver = BimanualIKSolver(model, data, left_ee_name, right_ee_name)

    # 启动 3D 渲染窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n" + "="*50)
        print(">>> MuJoCo 3D Viewer 已启动！")
        print(">>> 鼠标左键旋转视口，右键平移，滚轮缩放。")
        print(">>> 正在逐条完整物理回放...")
        print("="*50 + "\n")

        start_idx = 0
        for ep_i in range(num_episodes):
            end_idx = episode_ends[ep_i]
            ep_len = end_idx - start_idx
            print(f"▶ 正在回放第 [{ep_i + 1}/{num_episodes}] 条轨迹 (共 {ep_len} 步)...")

            # 1. 重置关节基础物理状态
            mujoco.mj_resetData(model, data)

            # 2. 严格对齐当前轨迹的 T 块与目标虚影初始位姿
            if has_t_init:
                t_init = t_init_poses[ep_i]
                t_jnt_adr = model.joint("tee_freejoint").qposadr[0]
                data.qpos[t_jnt_adr : t_jnt_adr + 7] = t_init

            if has_goal_init:
                g_init = goal_poses[ep_i]
                g_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
                model.body_pos[g_body_id] = g_init[:3]
                model.body_quat[g_body_id] = g_init[3:]

            # 推进动力学前向传播，更新空间几何变换
            mujoco.mj_forward(model, data)

            # 3. 逐帧执行动作并推进物理仿真
            for step_i in range(start_idx, end_idx):
                if not viewer.is_running():
                    print("Viewer 已关闭，退出回放。")
                    if HAS_CV2: cv2.destroyAllWindows()
                    return

                # 读取录制的左右手 3D 目标坐标
                act = actions[step_i]
                target_l = act[:3]
                target_r = act[3:]

                # IK 逆解关节控制量并执行仿真步进
                target_q = ik_solver.solve(target_l, target_r)
                data.ctrl[:len(target_q)] = target_q
                mujoco.mj_step(model, data)

                # 刷新 3D 画面
                viewer.sync()

                # 同步弹出当时采录的俯视实况相机画面
                if HAS_CV2 and has_imgs:
                    bgr_img = cv2.cvtColor(imgs[step_i], cv2.COLOR_RGB2BGR)
                    bgr_img_large = cv2.resize(bgr_img, (384, 384), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(bgr_img_large, f"Ep: {ep_i+1} | Step: {step_i - start_idx}/{ep_len}", 
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Recorded Camera View (Top)", bgr_img_large)
                    cv2.waitKey(1)

                # 控制回放帧率，保持动作自然连贯
                time.sleep(0.02)

            start_idx = end_idx
            time.sleep(0.6)  # 每条轨迹结束后短暂暂停

        print("\n所有轨迹动态回放完毕！按窗口右上角叉号或终端 Ctrl+C 退出。")
        if HAS_CV2: 
            cv2.destroyAllWindows()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zarr_file = os.path.join(current_dir, "push_t_demo.zarr")
    xml_file = os.path.join(current_dir, "openarm_push_t.xml")
    
    replay_dataset(zarr_path=zarr_file, xml_path=xml_file)
