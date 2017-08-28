
include_directories(
  model
)

add_executable(NNCNNAttLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNAttLabeler ${LIBS})
