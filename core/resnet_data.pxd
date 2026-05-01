"""Declaration file for dense ResNet state-vector extraction."""

from core.state cimport GameState


cpdef int get_resnet_vector_size(int num_players)


cpdef void get_resnet_data(GameState state, float[::1] buffer)


cpdef void get_resnet_data_batch(
    list state_arrays, int num_players, float[:, ::1] buffer,
)
