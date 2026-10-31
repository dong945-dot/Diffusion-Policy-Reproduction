import os
import time
import zarr
import numpy as np
import mujoco
import mujoco.viewer

def replay():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zarr_path = os.path.join(current_dir, "push_t_demo.zarr")
    xml_path = os.path.join(current_dir, "openarm_pusht.xml")

    print(f"正在加载数据集: {zarr_path} ...")
    root = zarr.open(zarr_path, mode='r')
    qposes = root['data/joint_qpos'][:]       # 真实关节角度 (N, 14)
    t_inits = root['data/t_init_pose'][:]
    g_inits = root['data/goal_pose'][:]
    ep_ends = root['meta/episode_ends'][:]

    # 💡 动态获取实际录制的手臂关节数量 (14维)
    num_arm_dofs = qposes.shape[1]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"\n>>> 启动绝对保真回放！成功加载 {len(ep_ends)} 条示教轨迹，每步控制 {num_arm_dofs} 个关节。")
        start_idx = 0
        for ep_i in range(len(ep_ends)):
            end_idx = ep_ends[ep_i]
            print(f"▶ 正在播放第 [{ep_i + 1}/{len(ep_ends)}] 条轨迹...")

            mujoco.mj_resetData(model, data)

            # 复原初始 T 块
            t_jnt_adr = model.joint("tee_freejoint").qposadr[0]
            data.qpos[t_jnt_adr : t_jnt_adr + 7] = t_inits[ep_i]

            # 复原目标虚影
            g_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_zone")
            model.body_pos[g_body_id] = g_inits[ep_i][:3]
            model.body_quat[g_body_id] = g_inits[ep_i][3:]

            # 初始机械臂赋角度 (严格按 num_arm_dofs 切片，避开 14 vs 16 报错)
            data.qpos[:num_arm_dofs] = qposes[start_idx]
            mujoco.mj_forward(model, data)

            for step_i in range(start_idx, end_idx):
                if not viewer.is_running():
                    return

                # 直接精准驱动 14 个手臂关节
                data.qpos[:num_arm_dofs] = qposes[step_i]
                data.ctrl[:num_arm_dofs] = qposes[step_i]
                mujoco.mj_step(model, data)

                viewer.sync()
                time.sleep(0.02)

            start_idx = end_idx
            time.sleep(0.5)

    print("\n所有轨迹动态回放完毕！")

if __name__ == "__main__":
    replay()
