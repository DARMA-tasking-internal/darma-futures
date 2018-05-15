

add_executable(pic pic.cc)
add_executable(picSendRecv picSendRecv.cc)
add_executable(stencil stencil.cc)

target_link_libraries(pic darma)
target_link_libraries(picSendRecv darma)
target_link_libraries(stencil darma)

