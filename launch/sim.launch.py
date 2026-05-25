import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Khóa chặt đường dẫn vào Desktop của bạn
    base_path = os.path.expanduser('~/Desktop/my_4ws_robot')
    xacro_file = os.path.join(base_path, 'urdf', 'car_4ws.urdf.xacro')
    world_file = os.path.join(base_path, 'worlds', 'maze.world')
    
    # Biên dịch URDF
    doc = xacro.process_file(xacro_file)
    robot_description = {'robot_description': doc.toxml()}

    # Khởi động Gazebo kèm theo file Map (maze.world)
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so', world_file], 
        output='screen'
    )

    # Thả xe vào môi trường
    spawn_entity = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_car'], 
        output='screen'
    )

    # Đăng ký trạng thái xe với ROS 2
    node_robot_state_publisher = Node(
        package='robot_state_publisher', 
        executable='robot_state_publisher',
        output='screen', 
        parameters=[robot_description]
    )

    # Kích hoạt lần lượt các hệ thống điều khiển
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'], 
        output='screen'
    )
    load_steering_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'steering_controller'], 
        output='screen'
    )
    load_wheel_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'wheel_controller'], 
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(event_handler=OnProcessExit(target_action=spawn_entity, on_exit=[load_joint_state_broadcaster])),
        RegisterEventHandler(event_handler=OnProcessExit(target_action=load_joint_state_broadcaster, on_exit=[load_steering_controller])),
        RegisterEventHandler(event_handler=OnProcessExit(target_action=load_steering_controller, on_exit=[load_wheel_controller])),
    ])