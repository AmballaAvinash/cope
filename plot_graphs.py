import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
plt.figure(figsize=(8, 5))

method = "qnli"
type = "train"


loss_rope= pd.read_csv(f"loss_{method}/{type}_loss_rope_glue.csv").values
plt.errorbar(
        range(1, loss_rope.shape[1]  + 1),
        np.mean(loss_rope, 0),
        yerr=np.std(loss_rope, 0),
        # fmt='o',
        label="ROPE Valued",
        color='darkorange',
        linestyle="-")

# loss_learned= pd.read_csv(f"loss_{method}/{type}_loss_learned_glue.csv").values
# plt.errorbar(
#     range(1, loss_learned.shape[1] + 1),
#     np.mean(loss_learned, 0),
#     yerr=np.std(loss_learned, 0),
#     fmt='^',
#     label="Learned Valued",
#     color='purple',
#     linestyle="--"
# )

# loss_sine= pd.read_csv(f"loss_{method}/{type}_loss_sine_glue.csv").values
# plt.errorbar(
#     range(1, loss_sine.shape[1] + 1),
#     np.mean(loss_sine, 0),
#     yerr=np.std(loss_sine, 0),
#     fmt='s',  # Square marker
#     label="Sine Valued",
#     color='green',
#     linestyle=":"
# )


# loss_complex_real= pd.read_csv(f"loss_{method}/{type}_loss_complex_glue_real.csv").values
# plt.errorbar(
#         range(1, loss_complex_real.shape[1]  + 1),
#         np.mean(loss_complex_real, 0),
#         yerr=np.std(loss_complex_real, 0),
#         # fmt='d',  # Diamond marker
#         label="Complex Valued real",
#         color='yellow',
#         linestyle="--")  # Dash-dot


# loss_complex_hybrid= pd.read_csv(f"loss_{method}/{type}_loss_complex_glue_hybrid.csv").values
# plt.errorbar(
#         range(1, loss_complex_hybrid.shape[1]  + 1),
#         np.mean(loss_complex_hybrid, 0),
#         yerr=np.std(loss_complex_hybrid, 0),
#         # fmt='d',  # Diamond marker
#         label="Complex Valued hybrid",
#         color='red',
#         linestyle="--")  # Dash-dot


loss_complex_phase= pd.read_csv(f"loss_{method}/{type}_loss_complex_glue_phase.csv").values
plt.errorbar(
        range(1, loss_complex_phase.shape[1]  + 1),
        np.mean(loss_complex_phase, 0),
        yerr=np.std(loss_complex_phase, 0),
        # fmt='d',  # Diamond marker
        label="Complex Valued phase",
        color='teal',
        linestyle="--")  # Dash-dot


loss_complex_magnitude= pd.read_csv(f"loss_{method}/{type}_loss_complex_glue_magnitude.csv").values
plt.errorbar(
        range(1, loss_complex_magnitude.shape[1]  + 1),
        np.mean(loss_complex_magnitude, 0),
        yerr=np.std(loss_complex_magnitude, 0),
        # fmt='d',  # Diamond marker
        label="Complex Valued magnitude",
        color='blue',
        linestyle="--")  # Dash-dot


loss_complex_hybrid_norm= pd.read_csv(f"loss_{method}/{type}_loss_complex_glue_hybrid_norm.csv").values
plt.errorbar(
        range(1, loss_complex_hybrid_norm.shape[1]  + 1),
        np.mean(loss_complex_hybrid_norm, 0),
        yerr=np.std(loss_complex_hybrid_norm, 0),
        # fmt='d',  # Diamond marker
        label="Complex Valued hybrid norm",
        color='black',
        linestyle="--")  # Dash-dot


plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epochs")
# plt.legend()
plt.tight_layout()
plt.grid(True)
plt.yticks() 
plt.xticks() 
plt.savefig(f"loss_{method}/{type}_{method}.pdf")