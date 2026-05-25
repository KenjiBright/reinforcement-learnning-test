import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    base_path = os.path.expanduser('~/Desktop/my_4ws_robot')
    xacro_file = os.path.join(base_path, 'urdf', 'car_4ws.urdf.xacro')
    # Trỏ trực tiếp tới bản đồ chữ L mới tạo
    world_file = os.path.join(base_path, 'worlds', 'maze_l_shape.world')
    
    doc = xacro.process_file(xacro_file)
    robot_description = {'robot_description': doc.toxml()}

    # Khởi động Gazebo cùng các plugin quản lý trạng thái hệ thống và bản đồ chữ L
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_state.so', world_file], 
        output='screen'
    )

    # Spawn mô hình robot my_car vào môi trường Gazebo từ dữ liệu robot_description
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py', arguments=['-topic', 'robot_description', '-entity', 'my_car', '-x', '0.5', '-y', '0.0', '-z', '0.05'], output='screen')
    node_robot_state_publisher = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen', parameters=[robot_description])

    # Khởi tạo các bộ điều khiển truyền động bánh xe và góc lái (Swerve Controllers) của hệ sinh thái ros2_control
    load_joint_state_broadcaster = ExecuteProcess(cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'], output='screen')
    load_steering_controller = ExecuteProcess(cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'steering_controller'], output='screen')
    load_wheel_controller = ExecuteProcess(cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'wheel_controller'], output='screen')

    # Quản lý vòng đời khởi động tuần tự tránh xung đột tài nguyên phần cứng ảo
    return LaunchDescription([
        gazebo, node_robot_state_publisher, spawn_entity,
        RegisterEventHandler(event_handler=OnProcessExit(target_action=spawn_entity, on_exit=[load_joint_state_broadcaster])),
        RegisterEventHandler(event_handler=OnProcessExit(target_action=load_joint_state_broadcaster, on_exit=[load_steering_controller])),
        RegisterEventHandler(event_handler=OnProcessExit(target_action=load_steering_controller, on_exit=[load_wheel_controller])),
    ])
