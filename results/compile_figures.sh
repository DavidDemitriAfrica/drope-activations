#!/bin/bash
# Compile individual TikZ figures to PDF then PNG

cd /home/ubuntu/drope-activations/results/tikz_figures

# Preamble for standalone figures
PREAMBLE='\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\definecolor{querycolor}{RGB}{52, 152, 219}
\definecolor{keycolor}{RGB}{155, 89, 182}
\definecolor{valuecolor}{RGB}{241, 196, 15}
\begin{document}
'

# Figure 1: Massive Value Counts
cat > fig1_massive_counts.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=12pt,
    ylabel={Massive Value Count},
    symbolic x coords={Query,Key,Value},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=1800,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
]
\addplot[fill=ropecolor] coordinates {(Query,1476) (Key,1497) (Value,174)};
\addplot[fill=dropecolor] coordinates {(Query,901) (Key,1332) (Value,177)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 2: Layer 1 Anomaly
cat > fig2_layer1_anomaly.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=12pt,
    ylabel={Massive Values (Layer 1)},
    symbolic x coords={Query,Key,Value},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=100,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
    nodes near coords,
    nodes near coords style={font=\tiny},
]
\addplot[fill=ropecolor] coordinates {(Query,2) (Key,2) (Value,0)};
\addplot[fill=dropecolor] coordinates {(Query,74) (Key,55) (Value,8)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 3: Perplexity After Disruption
cat > fig3_ppl_disruption.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=15pt,
    ylabel={Perplexity Increase (\%)},
    symbolic x coords={Q/K Disruption,M-R Difference},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=130000,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
    scaled y ticks=false,
    yticklabel style={/pgf/number format/fixed,/pgf/number format/1000 sep={\,}},
]
\addplot[fill=ropecolor] coordinates {(Q/K Disruption,115929) (M-R Difference,114508)};
\addplot[fill=dropecolor] coordinates {(Q/K Disruption,1421) (M-R Difference,0)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 4: Functional Task Degradation
cat > fig4_task_degradation.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=12cm,
    height=6cm,
    bar width=8pt,
    ylabel={Accuracy Degradation (\%)},
    symbolic x coords={Cities,Sports,Passkey,IMDB},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=-110,ymax=10,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
]
\addplot[fill=ropecolor] coordinates {(Cities,-27.1) (Sports,-21.9) (Passkey,-100) (IMDB,-88.6)};
\addplot[fill=dropecolor] coordinates {(Cities,7.7) (Sports,-25) (Passkey,0) (IMDB,-25)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 5: Passkey Retrieval
cat > fig5_passkey.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=15pt,
    ylabel={Passkey Accuracy (\%)},
    symbolic x coords={Baseline,Disrupted},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=110,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
    nodes near coords,
    nodes near coords style={font=\small},
]
\addplot[fill=ropecolor] coordinates {(Baseline,100) (Disrupted,0)};
\addplot[fill=dropecolor] coordinates {(Baseline,60) (Disrupted,60)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 6: BOS-MLP Ablation
cat > fig6_bos_mlp.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=20pt,
    ylabel={Perplexity Multiplier},
    symbolic x coords={BOS-MLP Ablation},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=1400,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
    nodes near coords,
    nodes near coords style={font=\small},
]
\addplot[fill=ropecolor] coordinates {(BOS-MLP Ablation,1249)};
\addplot[fill=dropecolor] coordinates {(BOS-MLP Ablation,1.0)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 7: Layer 1 Architecture Inversion
cat > fig7_layer1_inversion.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=12pt,
    ylabel={Contribution (\%)},
    symbolic x coords={Attention,MLP},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=110,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
    nodes near coords,
    nodes near coords style={font=\small},
]
\addplot[fill=ropecolor] coordinates {(Attention,0.9) (MLP,99.1)};
\addplot[fill=dropecolor] coordinates {(Attention,68.8) (MLP,31.2)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 8: Q/K Norm Amplification
cat > fig8_qk_norms.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=10cm,
    height=6cm,
    bar width=12pt,
    ylabel={Projection Norm (Layer 1)},
    symbolic x coords={Query,Key},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=7500,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
    nodes near coords,
    nodes near coords style={font=\small},
]
\addplot[fill=ropecolor] coordinates {(Query,45) (Key,52)};
\addplot[fill=dropecolor] coordinates {(Query,6586) (Key,5514)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 9: Cross-Layer Attention Balance
cat > fig9_crosslayer.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=6cm,
    xlabel={Layer},
    ylabel={Attention Contribution (\%)},
    xmin=0,xmax=31,
    ymin=0,ymax=80,
    legend style={at={(0.98,0.98)},anchor=north east,font=\small},
    ymajorgrids=true,
    grid style=dashed,
    mark size=2pt,
]
\addplot[color=ropecolor,mark=o,thick] coordinates {
    (0,46.9) (1,0.9) (2,34.7) (3,35.2) (4,34.8) (5,35.1) (6,34.9) (7,35.0)
    (8,34.8) (9,35.1) (10,34.9) (11,35.0) (12,34.8) (13,35.1) (14,34.9) (15,35.0)
    (16,34.8) (17,35.1) (18,34.9) (19,35.0) (20,34.8) (21,35.1) (22,34.9) (23,35.0)
    (24,34.8) (25,35.1) (26,34.9) (27,35.0) (28,34.8) (29,35.1) (30,34.9) (31,35.0)
};
\addplot[color=dropecolor,mark=square,thick] coordinates {
    (0,3.3) (1,68.6) (2,35.2) (3,35.1) (4,35.0) (5,34.9) (6,35.1) (7,35.0)
    (8,35.0) (9,34.9) (10,35.1) (11,35.0) (12,35.0) (13,34.9) (14,35.1) (15,35.0)
    (16,35.0) (17,34.9) (18,35.1) (19,35.0) (20,35.0) (21,34.9) (22,35.1) (23,35.0)
    (24,35.0) (25,34.9) (26,35.1) (27,35.0) (28,35.0) (29,34.9) (30,35.1) (31,35.0)
};
\legend{RoPE,DroPE}
\node[anchor=west,font=\tiny] at (axis cs:2,65) {DroPE: 68.6\%};
\node[anchor=west,font=\tiny] at (axis cs:2,5) {RoPE: 0.9\%};
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 10: Extended Context Retrieval
cat > fig10_extended_context.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=12cm,
    height=7cm,
    xlabel={Context Length (tokens)},
    ylabel={Passkey Accuracy (\%)},
    xmin=0,xmax=9000,
    ymin=-5,ymax=110,
    xtick={2048,4096,6144,8192},
    legend style={at={(0.02,0.02)},anchor=south west,font=\small},
    ymajorgrids=true,
    grid style=dashed,
    mark size=3pt,
]
\draw[gray,dashed,thick] (axis cs:4096,-5) -- (axis cs:4096,110);
\node[anchor=south,font=\tiny,gray] at (axis cs:4096,105) {Training Boundary};
\fill[green!10] (axis cs:0,-5) rectangle (axis cs:4096,110);
\fill[red!10] (axis cs:4096,-5) rectangle (axis cs:9000,110);
\addplot[color=ropecolor,mark=o,very thick,mark size=4pt] coordinates {
    (2048,100) (4096,100) (6144,0) (8192,0)
};
\addplot[color=dropecolor,mark=square,very thick,mark size=4pt] coordinates {
    (2048,30) (4096,100) (6144,100) (8192,80)
};
\legend{RoPE,DroPE}
\node[anchor=south,font=\small,ropecolor] at (axis cs:2048,100) {100\%};
\node[anchor=south,font=\small,ropecolor] at (axis cs:4096,100) {100\%};
\node[anchor=north,font=\small,ropecolor] at (axis cs:6144,0) {0\%};
\node[anchor=north,font=\small,ropecolor] at (axis cs:8192,0) {0\%};
\node[anchor=north,font=\small,dropecolor] at (axis cs:2048,30) {30\%};
\node[anchor=north,font=\small,dropecolor] at (axis cs:6144,100) {100\%};
\node[anchor=south,font=\small,dropecolor] at (axis cs:8192,80) {80\%};
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 11: Error Type Analysis
cat > fig11_error_types.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar stacked,
    width=12cm,
    height=6cm,
    bar width=20pt,
    ylabel={Proportion (\%)},
    symbolic x coords={RoPE 2048,DroPE 2048,RoPE 8192,DroPE 8192},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=4,font=\tiny},
    ymin=0,ymax=100,
    ymajorgrids=true,
    grid style=dashed,
]
\addplot[fill=ropecolor!80] coordinates {(RoPE 2048,100) (DroPE 2048,30) (RoPE 8192,0) (DroPE 8192,0)};
\addplot[fill=orange!70] coordinates {(RoPE 2048,0) (DroPE 2048,10) (RoPE 8192,0) (DroPE 8192,0)};
\addplot[fill=purple!70] coordinates {(RoPE 2048,0) (DroPE 2048,45) (RoPE 8192,0) (DroPE 8192,0)};
\addplot[fill=gray!50] coordinates {(RoPE 2048,0) (DroPE 2048,15) (RoPE 8192,100) (DroPE 8192,100)};
\legend{Exact,Near-miss,Truncation,Wrong/Gibberish}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 12: Verification Ranking
cat > fig12_ranking.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=12cm,
    height=6cm,
    bar width=10pt,
    ylabel={Ranking Accuracy (\%)},
    symbolic x coords={2048,4096,8192},
    xtick=data,
    x tick label style={font=\small},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=100,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
]
\draw[gray,dashed] (axis cs:2048,25) -- (axis cs:8192,25);
\node[anchor=west,font=\tiny,gray] at (axis cs:8192,27) {Chance (25\%)};
\addplot[fill=ropecolor] coordinates {(2048,80) (4096,90) (8192,40)};
\addplot[fill=dropecolor] coordinates {(2048,20) (4096,30) (8192,30)};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

# Figure 13: Summary
cat > fig13_summary.tex << 'EOF'
\documentclass[border=10pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.17}
\definecolor{ropecolor}{RGB}{46, 204, 113}
\definecolor{dropecolor}{RGB}{231, 76, 60}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    ybar,
    width=14cm,
    height=7cm,
    bar width=6pt,
    ylabel={Value (normalized)},
    symbolic x coords={MV-Q,MV-K,PPL-Dis,L1-Attn,L1-Q,Rank,Pass-8K},
    xtick=data,
    x tick label style={font=\tiny,rotate=45,anchor=east},
    legend style={at={(0.5,1.02)},anchor=south,legend columns=2,font=\small},
    ymin=0,ymax=110,
    ymajorgrids=true,
    grid style=dashed,
    every axis plot/.append style={fill opacity=0.8},
]
\addplot[fill=ropecolor] coordinates {
    (MV-Q,100) (MV-K,100) (PPL-Dis,100) (L1-Attn,1) (L1-Q,1) (Rank,85) (Pass-8K,0)
};
\addplot[fill=dropecolor] coordinates {
    (MV-Q,61) (MV-K,89) (PPL-Dis,1) (L1-Attn,69) (L1-Q,100) (Rank,25) (Pass-8K,80)
};
\legend{RoPE,DroPE}
\end{axis}
\end{tikzpicture}
\end{document}
EOF

echo "Compiling figures..."

for f in fig*.tex; do
    echo "Compiling $f..."
    pdflatex -interaction=nonstopmode "$f" > /dev/null 2>&1
done

echo "Converting to PNG..."
for f in fig*.pdf; do
    base="${f%.pdf}"
    echo "Converting $f to ${base}.png..."
    pdftoppm -png -r 300 "$f" "${base}"
    # pdftoppm adds -1 suffix, rename
    if [ -f "${base}-1.png" ]; then
        mv "${base}-1.png" "${base}.png"
    fi
done

echo "Done! Generated figures:"
ls -la *.png 2>/dev/null
