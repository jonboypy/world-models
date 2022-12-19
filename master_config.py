
class MasterConfig:

    # (int) The size of the latent vector Z
    Z_SIZE = 32

    # (int) The size of the LSTM hidden vector
    HX_SIZE = 256

    # (int) The size of the action space
    ACTION_SPACE_SIZE = 3

    # (int) The number of Gaussians to model for mixture density network
    N_GAUSSIANS = 5

    # (bool) Determines if LSTM cell state is concatenated with z
    LSTM_CELL_ST = False

    # (float) temperature parameter that controls the model uncertainty
    TEMP = 1.0
