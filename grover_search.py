
# Set matplotlib backend for headless environments (Streamlit Cloud, etc.)
import matplotlib
matplotlib.use("Agg")

import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import io
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import DiagonalGate

st.set_page_config(page_title="Grover's Algorithm Demo (Extended)", layout="wide")

# ---------------------------
# Helper functions & gates
# ---------------------------

def human_to_index(bitstr: str) -> int:
    """Convert human MSB-left bitstring to Qiskit's little-endian index."""
    return int(bitstr[::-1], 2)

def index_to_human(idx: int, n: int) -> str:
    """Convert Qiskit index to human MSB-left bitstring."""
    return format(idx, f'0{n}b')

@st.cache_data
def make_oracle_gate(marked_states, n_qubits):
    """Build a Diagonal oracle gate that flips phase (-1) of marked states."""
    size = 2 ** n_qubits
    diag = [1.0] * size
    for s in marked_states:
        idx = human_to_index(s)
        diag[idx] = -1.0
    diag_gate = DiagonalGate(diag)
    diag_gate.label = "Oracle"
    return diag_gate

@st.cache_data
def make_diffusion_gate(n_qubits):
    """Build diffusion (inversion about average) as a single gate (H^n · Phase0 · H^n)."""
    size = 2 ** n_qubits
    diag_phase0 = [-1.0] + [1.0] * (size - 1)
    phase0_gate = DiagonalGate(diag_phase0)
    phase0_gate.label = "Phase0"
    qc = QuantumCircuit(n_qubits, name="Diffusion")
    qc.h(range(n_qubits))
    qc.append(phase0_gate, range(n_qubits))
    qc.h(range(n_qubits))
    return qc.to_gate()

def probs_from_statevector(sv: Statevector, n_qubits: int):
    """Map statevector to dict of human bitstrings -> probabilities."""
    probs = {index_to_human(i, n_qubits)[::-1]: 0.0 for i in range(2 ** n_qubits)}
    for idx, amp in enumerate(sv.data):
        human = index_to_human(idx, n_qubits)[::-1]
        probs[human] += np.abs(amp) ** 2
    return probs

def counts_to_human_counts(raw_counts, n_qubits):
    """Normalize raw simulator counts into human-ordered bitstrings."""
    human_states = [index_to_human(i, n_qubits) for i in range(2 ** n_qubits)]
    counts = {s: 0 for s in human_states}
    for key, val in raw_counts.items():
        k = key.replace(" ", "")
        if len(k) != n_qubits:
            k = k.split()[0]
        k_human = k[::-1]  # Convert Qiskit little-endian to human MSB-left
        if k_human in counts:
            counts[k_human] += val
        else:
            counts[human_states[0]] += val
    return counts

def qiskit_circuit_to_png_bytes(qc, scale=1.0):
    """
    Draw a qiskit circuit using matplotlib and return PNG bytes.
    The draw call returns a matplotlib figure which we can save.
    """
    try:
        fig = qc.draw(output="mpl", scale=scale)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        import traceback
        print("Circuit PNG generation failed:", e)
        print(traceback.format_exc())
        return b""

# ---------------------------
# UI Controls
# ---------------------------

st.title("🔎 Grover's Search — Interactive Teaching App (Extended)")

st.write(
    "This app simulates Grover's algorithm step-by-step, supports multiple marked states, "
    "lets you scrub through amplitude evolution, and provides export/download options.\n\n"
    "**Bitstring convention:** enter bitstrings MSB-left (human-friendly). Example for 3 qubits: '001'."
)

with st.sidebar:
    st.header("Simulation Controls")
    num_qubits = st.slider("Number of qubits", min_value=2, max_value=5, value=3, step=1)
    default_target = "0" * (num_qubits - 1) + "1"
    target_input = st.text_input("Marked target states (comma separated)", default_target)
    shots = st.slider("Shots (measurement simulation)", min_value=256, max_value=16384, value=2048, step=256)
    show_statevectors = st.checkbox("Show statevector & probability tables", value=True)
    show_circuits = st.checkbox("Show circuits (visual)", value=True)
    autostep_speed = st.slider("Animation speed (sec per step for Play)", 0.05, 1.0, 0.25, 0.05)
    run_button = st.button("Run Grover Demonstration ▶")

# parse and validate targets
raw_targets = [t.strip() for t in target_input.split(",") if t.strip() != ""]
valid = True
if len(raw_targets) == 0:
    valid = False
    st.sidebar.error("Provide at least one target bitstring.")
else:
    for t in raw_targets:
        if len(t) != num_qubits or any(c not in "01" for c in t):
            valid = False
            st.sidebar.error(f"Invalid bitstring: '{t}'. Must be {num_qubits} bits, only 0/1.")

if not valid:
    st.stop()

marked_states = raw_targets
num_marked = len(marked_states)
num_total = 2 ** num_qubits

st.subheader("Search Problem")
st.markdown(
    f"- Number of qubits: **{num_qubits}**  \n"
    f"- Total possible states: **{num_total}**  \n"
    f"- Marked (target) states: **{', '.join(marked_states)}**  \n"
    f"- Classical random success probability: **{num_marked}/{num_total} = {num_marked/num_total:.3f}**"
)

# compute suggested optimal iterations
def optimal_iterations(num_marked, num_total):
    if num_marked == 0:
        return 0
    theta = math.asin(math.sqrt(num_marked / num_total))
    if theta == 0:
        return 0
    opt = math.pi / (4 * theta)
    return max(1, math.floor(opt))

opt_iters = optimal_iterations(num_marked, num_total)
st.info(f"Suggested (optimal) number of Grover iterations: **{opt_iters}**")

if not run_button and "initialized" not in st.session_state:
    st.write("Press **Run Grover Demonstration ▶** in the sidebar to start the simulation.")
    st.stop()

# ---------------------------
# Run / (re)use stored simulation
# ---------------------------

def run_and_store():
    """Run the full simulation once and store results into session_state."""
    oracle_gate = make_oracle_gate(marked_states, num_qubits)
    diffusion_gate = make_diffusion_gate(num_qubits)

    # Build starting circuit (H^n)
    qc_init = QuantumCircuit(num_qubits)
    qc_init.h(range(num_qubits))
    # Collect step labels & statevectors
    step_labels = []
    sv_list = []

    # after H
    step_labels.append("After Hadamard (equal superposition)")
    sv_list.append(Statevector.from_instruction(qc_init))

    # after first oracle
    qc_after_oracle = qc_init.copy()
    qc_after_oracle.append(oracle_gate, range(num_qubits))
    step_labels.append(f"After Oracle (marks {', '.join(marked_states)})")
    sv_list.append(Statevector.from_instruction(qc_after_oracle))

    # after first diffusion (one full iteration)
    qc_after_diff = qc_after_oracle.copy()
    qc_after_diff.append(diffusion_gate, range(num_qubits))
    step_labels.append("After Diffusion (one Grover iteration complete)")
    sv_list.append(Statevector.from_instruction(qc_after_diff))

    # additional iterations
    current_qc = qc_after_diff.copy()
    for it in range(2, opt_iters + 1):
        current_qc.append(oracle_gate, range(num_qubits))
        sv_list.append(Statevector.from_instruction(current_qc))
        step_labels.append(f"After Oracle (iteration {it})")
        current_qc.append(diffusion_gate, range(num_qubits))
        sv_list.append(Statevector.from_instruction(current_qc))
        step_labels.append(f"After Diffusion (iteration {it})")

    # probabilities per step
    probs_per_step = [probs_from_statevector(sv, num_qubits) for sv in sv_list]

    # Build full measured circuit
    full_qc = QuantumCircuit(num_qubits)
    full_qc.h(range(num_qubits))
    for _ in range(opt_iters):
        full_qc.append(oracle_gate, range(num_qubits))
        full_qc.append(diffusion_gate, range(num_qubits))
    full_qc.measure_all()

    # Run shot-based simulation and get counts
    simulator_qasm = AerSimulator()
    transpiled = transpile(full_qc, simulator_qasm)
    job = simulator_qasm.run(transpiled, shots=shots)
    result = job.result()
    raw_counts = result.get_counts()
    human_counts = counts_to_human_counts(raw_counts, num_qubits)

    # Prepare PNG bytes for full circuit and oracle small circuit
    png_full = qiskit_circuit_to_png_bytes(full_qc, scale=1.0)
    qc_oracle_only = QuantumCircuit(num_qubits)
    qc_oracle_only.append(oracle_gate, range(num_qubits))
    png_oracle = qiskit_circuit_to_png_bytes(qc_oracle_only, scale=1.2)

    # Store everything in session_state
    st.session_state["initialized"] = True
    st.session_state["oracle_gate"] = oracle_gate
    st.session_state["diffusion_gate"] = diffusion_gate
    st.session_state["step_labels"] = step_labels
    st.session_state["sv_list"] = sv_list
    st.session_state["probs_per_step"] = probs_per_step
    st.session_state["full_qc"] = full_qc
    st.session_state["counts"] = human_counts
    st.session_state["png_full"] = png_full
    st.session_state["png_oracle"] = png_oracle
    st.session_state["shots"] = shots
    st.session_state["num_qubits"] = num_qubits
    st.session_state["marked_states"] = marked_states
    st.session_state["opt_iters"] = opt_iters

# If user just clicked Run, or we haven't initialized yet, run & store
if run_button or "initialized" not in st.session_state:
    run_and_store()

# Now everything is available in st.session_state — use it below without recomputing
oracle_gate = st.session_state["oracle_gate"]
diffusion_gate = st.session_state["diffusion_gate"]
step_labels = st.session_state["step_labels"]
sv_list = st.session_state["sv_list"]
probs_per_step = st.session_state["probs_per_step"]
iterations_count = len(probs_per_step)
full_qc = st.session_state["full_qc"]
human_counts = st.session_state["counts"]
png_full = st.session_state["png_full"]
png_oracle = st.session_state["png_oracle"]
shots = st.session_state["shots"]

# ---------------------------
# Step-by-step display
# ---------------------------

st.header("Step-by-step Statevectors & Probabilities (Interactive)")

for idx, (label, sv) in enumerate(zip(step_labels, sv_list)):
    with st.expander(f"{label}", expanded=False):
        probs = probs_per_step[idx]
        df = pd.DataFrame([{"state": s, "probability": p, "percentage": p * 100} for s, p in sorted(probs.items())])
        df["probability"] = df["probability"].round(6)
        df["percentage"] = df["percentage"].round(2)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("State probabilities (table)")
            st.dataframe(df.set_index("state"), width='stretch')
        with col2:
            st.subheader("Probability distribution (plot)")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors = ['red' if s in marked_states else 'C0' for s in df["state"]]
            bars = ax.bar(df["state"], df["probability"], color=colors, edgecolor='black', alpha=0.9)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("State")
            for i, row in df.iterrows():
                ax.text(i, row["probability"] + 0.01, f'{row["percentage"]:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.set_title(label)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        if show_circuits:
            st.subheader("Circuit (visual)")
            # Build minimal circuit that reaches this step (for display only)
            temp_qc = QuantumCircuit(num_qubits)
            temp_qc.h(range(num_qubits))
            # apply steps until matching label
            for j in range(1, iterations_count):
                if "Oracle" in step_labels[j]:
                    temp_qc.append(oracle_gate, range(num_qubits))
                elif "Diffusion" in step_labels[j]:
                    temp_qc.append(diffusion_gate, range(num_qubits))
                if step_labels[j] == label:
                    break
            try:
                fig2 = temp_qc.draw(output="mpl", scale=1.0)
                st.pyplot(fig2)
            except Exception as e:
                st.text(f"Circuit rendering not available in this environment. Error: {e}")

# ---------------------------
# Amplitude evolution animation (slider + play)
# ---------------------------

st.header("Amplitude evolution across steps (interactive slider)")
st.write("Use the slider to scrub through the precomputed steps. Marked states are highlighted in red.")

state_order = sorted(list(probs_per_step[0].keys()))
# Slider to choose step
step_idx = st.slider("Choose step to view", min_value=0, max_value=iterations_count - 1, value=iterations_count - 1, step=1)

# Play / Stop buttons using session_state flag
if "playing" not in st.session_state:
    st.session_state["playing"] = False

play_clicked = st.button("Play animation ▶")
stop_clicked = st.button("Stop")

if play_clicked:
    st.session_state["playing"] = True
if stop_clicked:
    st.session_state["playing"] = False

def render_step_plot(i):
    selected_probs = probs_per_step[i]
    df_sel = pd.DataFrame([{"state": s, "probability": selected_probs[s], "percentage": selected_probs[s] * 100} for s in state_order])
    fig_sel, ax_sel = plt.subplots(figsize=(10, 3.5))
    ax_sel.bar(df_sel["state"], df_sel["probability"], color=['red' if s in marked_states else 'C0' for s in df_sel["state"]],
               edgecolor='black', alpha=0.9)
    ax_sel.set_ylim(0, 1.0)
    ax_sel.set_ylabel("Probability")
    ax_sel.set_xlabel("State")
    ax_sel.set_title(f"Step {i}: {step_labels[i]}")
    plt.xticks(rotation=45)
    for idx_row, row in df_sel.iterrows():
        ax_sel.text(idx_row, row["probability"] + 0.01, f'{row["percentage"]:.1f}%', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig_sel)

# Initial render of selected step
render_step_plot(step_idx)

# If playing, loop through remaining steps (non-blocking render; user can Stop)
if st.session_state["playing"]:
    for i in range(step_idx, iterations_count):
        if not st.session_state["playing"]:
            break
        render_step_plot(i)
        time.sleep(autostep_speed)
    # stop automatically when done
    st.session_state["playing"] = False

# ---------------------------
# Measurement-based simulation & export options
# ---------------------------

st.header("Measurement-based Simulation (shot-based) & Export")
st.write(f"We ran the full Grover circuit with **{opt_iters}** iterations and measured all qubits (shots = {shots}).")

if show_circuits:
    st.subheader("Full Grover Circuit (measured)")
    try:
        fig_full = full_qc.draw(output="mpl", scale=1.0)
        st.pyplot(fig_full)
    except Exception as e:
        st.text(f"Circuit rendering not available in this environment. Error: {e}")

# Map to human counts (already mapped)
total_shots = sum(human_counts.values())
target_shots = sum(human_counts.get(s, 0) for s in marked_states)
success_pct = target_shots / total_shots * 100 if total_shots > 0 else 0.0

# Counts dataframe
counts_df = pd.DataFrame([
    {"state": s, "counts": human_counts[s], "percentage": human_counts[s] / total_shots * 100}
    for s in sorted(human_counts.keys())
])
counts_df["percentage"] = counts_df["percentage"].round(2)
counts_df = counts_df.set_index("state")

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Measurement counts (human order)")
    st.dataframe(counts_df, width='stretch')
    # CSV download (uses stored counts)
    csv_bytes = counts_df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download counts CSV", csv_bytes, file_name="grover_counts.csv", mime="text/csv")
with colB:
    st.subheader("Measurement histogram")
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    ax3.bar(counts_df.index, counts_df["counts"], color=['red' if s in marked_states else 'C0' for s in counts_df.index], edgecolor='black', alpha=0.9)
    ax3.set_ylabel("Counts")
    ax3.set_xlabel("State")
    ax3.set_title(f"Measurement results — {shots} shots")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

st.success(f"🎯 Measured success probability for targets {', '.join(marked_states)}: **{success_pct:.2f}%**")
st.info(f"Classical random baseline: {(num_marked / num_total) * 100:.2f}%")

# ---------------------------
# Export circuit PNGs (full circuit & oracle)
# ---------------------------

st.header("Export circuit diagrams")

if png_full:
    st.download_button("⬇️ Download full Grover circuit PNG", png_full, "grover_full_circuit.png", "image/png")
else:
    st.text("Full circuit PNG generation failed in this environment.")

if png_oracle:
    st.download_button("⬇️ Download oracle circuit PNG", png_oracle, "grover_oracle.png", "image/png")
else:
    st.text("Oracle PNG generation failed in this environment.")

# ---------------------------
# Theory & tutorial
# ---------------------------

st.header("📘 Tutorial & Explanation (Oracle / Diffuser)")

with st.expander("What is the Oracle?"):
    st.markdown(
        "- The **oracle** flips the phase (multiplies by -1) of the marked state(s). This is a phase kick that does not change probabilities directly but changes amplitudes' signs so the diffusion step can amplify the target.\n"
        "- We implemented the oracle as a **diagonal gate** with -1 on marked entries and +1 elsewhere, which is easy to construct in simulation.\n"
        "- Example: for target `'001'` we flip the amplitude sign of the state `|001⟩`."
    )
    st.write("Oracle matrix (diagonal entries):")
    try:
        # Reconstruct the diagonal directly as in make_oracle_gate
        size = 2 ** num_qubits
        diag = [1.0] * size
        # For display, show both Qiskit index and human bitstring (MSB-left)
        human_bitstrings = [format(i, f'0{num_qubits}b') for i in range(size)]
        # Flip at the index matching the human bitstring (no reverse)
        for s in marked_states:
            idx = int(s, 2)
            diag[idx] = -1.0
        diag_df = pd.DataFrame({
            "index": range(size),
            "human_bitstring": human_bitstrings,
            "diag": diag
        })
        st.dataframe(diag_df.head(16), width='stretch')
    except Exception as e:
        st.text(f"Cannot display oracle diagonal matrix here. Error: {type(e).__name__}: {e}")

with st.expander("What is the Diffuser (Inversion-about-the-average)?"):
    st.markdown(
        "- The diffuser reflects amplitudes about the average amplitude, increasing amplitudes of marked states and decreasing others when applied after the oracle.\n"
        "- Implemented as H^n · (2|0⟩⟨0| - I) · H^n. We built it as a small composite gate using a diagonal phase that flips |0...0⟩ and H layers.\n"
        "- Intuition: Oracle does a phase flip on targets → diffuser translates that phase information into amplitude increases for the target states."
    )

st.write("---")
import imageio
from fpdf import FPDF

# ---------------------------
# Additional Exports: Per-iteration CSV, GIF, PDF
# ---------------------------

st.header("📂 Advanced Exports & Tutorial")

# 1️⃣ Export per-iteration amplitudes CSV
st.subheader("Per-iteration amplitudes CSV")
amplitude_rows = []
for i, probs in enumerate(st.session_state["probs_per_step"]):
    for state, p in probs.items():
        amplitude_rows.append({"step": i, "state": state, "probability": p})
df_amplitudes = pd.DataFrame(amplitude_rows)
csv_bytes_amp = df_amplitudes.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download amplitudes CSV", csv_bytes_amp, "grover_amplitudes.csv", "text/csv")

# 2️⃣ Animated GIF of amplitude evolution
st.subheader("Animated amplitude evolution (GIF)")

if st.button("Generate Animated GIF"):
    frames = []
    state_order = sorted(st.session_state["probs_per_step"][0].keys())
    for i, probs in enumerate(st.session_state["probs_per_step"]):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        values = [probs[s] for s in state_order]
        colors = ['red' if s in st.session_state["marked_states"] else 'C0' for s in state_order]
        ax.bar(state_order, values, color=colors, edgecolor='black', alpha=0.9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_xlabel("State")
        ax.set_title(f"Step {i}: {st.session_state['step_labels'][i]}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        plt.close(fig)
    gif_bytes = io.BytesIO()
    imageio.mimsave(gif_bytes, frames, format='GIF', duration=0.5)
    gif_bytes.seek(0)
    st.image(gif_bytes, caption="Grover amplitude evolution GIF")
    st.download_button("⬇️ Download GIF", gif_bytes, "grover_evolution.gif", "image/gif")

# ---------------------------
# PDF Tutorial Export (fpdf2 + Unicode font)
# ---------------------------
from fpdf import FPDF
import io
import os

st.header("📄 Download PDF Tutorial (Unicode Support)")

if st.button("Generate PDF Tutorial"):

    pdf = FPDF()
    pdf.add_page()
    
    # Add a TTF Unicode font
    # You need to have this font file in your project folder or specify a system path
    # Download from: https://www.fontsquirrel.com/fonts/dejavu-sans
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        st.error("Please add DejaVuSans.ttf to your project folder for Unicode PDF support.")
        st.stop()
    
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)
    
    # Title
    pdf.set_font("DejaVu", 'B', 16)
    pdf.multi_cell(0, 10, "Grover's Algorithm — Interactive Tutorial", align='C')
    
    pdf.ln(5)
    
    # Overview
    pdf.set_font("DejaVu", '', 12)
    pdf.multi_cell(0, 8,
                   f"Number of qubits: {num_qubits}\n"
                   f"Marked states: {', '.join(marked_states)}\n"
                   f"Optimal Grover iterations: {opt_iters}\n"
                   f"Shots used in measurement simulation: {shots}\n")
    
    pdf.ln(5)
    
    # Explanation
    pdf.multi_cell(0, 8,
                   "Explanation of Oracle and Diffuser:\n"
                   "- Oracle: flips the phase (−1) of the marked states, enabling amplitude amplification.\n"
                   "- Diffuser: inversion-about-average to amplify marked states after oracle application.\n"
                   "- Together, repeated iterations increase probability of measuring marked states.")
    
    pdf.ln(5)
    
    # Optional: Embed full circuit image
    try:
        if png_full:
            pdf.image(io.BytesIO(st.session_state["png_full"]), x=10, y=None, w=180)
    except Exception:
        pdf.multi_cell(0, 8, "Circuit image could not be embedded.")

    # Generate PDF bytes
    pdf_bytes = bytes(pdf.output(dest='S'))  # convert bytearray -> bytes
    st.download_button(
    "⬇️ Download PDF Tutorial",
    pdf_bytes,
    file_name="grover_tutorial.pdf",
    mime="application/pdf"
)