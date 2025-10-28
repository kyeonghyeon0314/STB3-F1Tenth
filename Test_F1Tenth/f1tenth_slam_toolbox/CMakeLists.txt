cmake_minimum_required(VERSION 3.8)
project(f1tenth_slam_toolbox)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# 설정 파일 설치
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/
)

# 런치 파일 설치
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Python 실행 파일 설치
install(PROGRAMS
  launch/scan_header_fix.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()