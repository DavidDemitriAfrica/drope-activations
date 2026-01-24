"""
Manim animation showing REAL attention patterns from RoPE vs DroPE.
This visualizes actual model internals extracted via forward hooks.
"""

from manim import *
import json
import numpy as np

# Load real data
with open("results/attention_animation_data.json", "r") as f:
    data = json.load(f)

rope_data = data["rope"]
drope_data = data["drope"]
tokens = rope_data["tokens"][:12]  # First 12 tokens for clarity


class AttentionFlowComparison(Scene):
    """
    Shows real attention patterns flowing through layers.
    Left: RoPE, Right: DroPE
    Each frame shows attention from one layer.
    """

    def construct(self):
        # Title
        title = Text("Actual Attention Patterns: RoPE vs DroPE", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Create token labels
        n_tokens = len(tokens)

        # Left side: RoPE
        rope_label = Text("RoPE (Llama-2-7B)", font_size=24, color=GREEN)
        rope_label.move_to(LEFT * 3.5 + UP * 2.5)

        # Right side: DroPE
        drope_label = Text("DroPE", font_size=24, color=RED)
        drope_label.move_to(RIGHT * 3.5 + UP * 2.5)

        self.play(Write(rope_label), Write(drope_label))

        # Create attention matrices for both models
        rope_matrices = []
        drope_matrices = []

        # We'll show layers 0, 1, 2, 8, 16, 31
        layers_to_show = [0, 1, 2, 8, 16, 31]

        for layer_idx in layers_to_show:
            # Get actual attention data
            rope_attn = np.array(rope_data["layers"][str(layer_idx)]["avg_attention"])[:n_tokens, :n_tokens]
            drope_attn = np.array(drope_data["layers"][str(layer_idx)]["avg_attention"])[:n_tokens, :n_tokens]

            # Create heatmap visualization
            rope_mat = self.create_attention_matrix(rope_attn, LEFT * 3.5)
            drope_mat = self.create_attention_matrix(drope_attn, RIGHT * 3.5)

            rope_matrices.append(rope_mat)
            drope_matrices.append(drope_mat)

        # Layer indicator
        layer_text = Text(f"Layer 0", font_size=28)
        layer_text.to_edge(DOWN, buff=0.5)

        # Q/K norm indicator
        qk_text = Text("", font_size=20)
        qk_text.next_to(layer_text, UP, buff=0.3)

        # Show first layer
        self.play(
            Create(rope_matrices[0]),
            Create(drope_matrices[0]),
            Write(layer_text)
        )
        self.wait(1)

        # Animate through layers
        for i, layer_idx in enumerate(layers_to_show[1:], 1):
            # Update layer text
            new_layer_text = Text(f"Layer {layer_idx}", font_size=28)
            new_layer_text.to_edge(DOWN, buff=0.5)

            # Get Q/K norms for this layer
            rope_q = rope_data["qk_norms"]["q"][layer_idx]
            drope_q = drope_data["qk_norms"]["q"][layer_idx]

            # Special highlight for Layer 1
            if layer_idx == 1:
                highlight_text = Text(
                    f"Layer 1: DroPE Q norm = {drope_q:.0f} (vs RoPE {rope_q:.0f}) → {drope_q/rope_q:.0f}× amplification",
                    font_size=18,
                    color=YELLOW
                )
                highlight_text.next_to(new_layer_text, UP, buff=0.3)

                self.play(
                    Transform(rope_matrices[i-1], rope_matrices[i]),
                    Transform(drope_matrices[i-1], drope_matrices[i]),
                    Transform(layer_text, new_layer_text),
                    Write(highlight_text),
                    run_time=1.5
                )
                self.wait(2)
                self.play(FadeOut(highlight_text))
            else:
                self.play(
                    Transform(rope_matrices[i-1], rope_matrices[i]),
                    Transform(drope_matrices[i-1], drope_matrices[i]),
                    Transform(layer_text, new_layer_text),
                    run_time=1
                )
                self.wait(0.5)

        self.wait(2)

    def create_attention_matrix(self, attn_weights, center_pos):
        """Create a visual representation of attention matrix."""
        n = attn_weights.shape[0]
        cell_size = 0.35

        group = VGroup()

        for i in range(n):
            for j in range(n):
                if j <= i:  # Only show causal attention
                    weight = attn_weights[i, j]
                    # Color intensity based on attention weight
                    color = interpolate_color(BLACK, WHITE, min(weight * 2, 1))
                    cell = Square(side_length=cell_size, fill_color=color, fill_opacity=0.9, stroke_width=0.5)
                    cell.move_to(center_pos + RIGHT * (j - n/2) * cell_size + DOWN * (i - n/2) * cell_size)
                    group.add(cell)

        return group


class Layer1Amplification(Scene):
    """
    Focused animation on Layer 1 Q/K amplification.
    Shows the actual norms as bar heights.
    """

    def construct(self):
        title = Text("Layer 1: How DroPE Makes Attention Matter", font_size=32)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Get actual Q/K norms for all layers
        rope_q = rope_data["qk_norms"]["q"]
        drope_q = drope_data["qk_norms"]["q"]

        # Create axes
        axes = Axes(
            x_range=[0, 32, 4],
            y_range=[0, 30000, 5000],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": list(range(0, 33, 4))},
            y_axis_config={"numbers_to_include": [0, 10000, 20000, 30000]},
        )
        axes.center().shift(DOWN * 0.5)

        x_label = Text("Layer", font_size=20).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Q Projection Norm", font_size=20).next_to(axes.y_axis, LEFT, buff=0.3).rotate(90*DEGREES)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # RoPE line (stays flat)
        rope_points = [axes.c2p(i, min(rope_q[i], 29000)) for i in range(32)]
        rope_line = VMobject(color=GREEN, stroke_width=3)
        rope_line.set_points_smoothly(rope_points)

        rope_legend = Text("RoPE", font_size=20, color=GREEN)
        rope_legend.move_to(axes.c2p(28, 5000))

        self.play(Create(rope_line), Write(rope_legend))
        self.wait(0.5)

        # DroPE line (massive spike at layer 1)
        drope_points = [axes.c2p(i, min(drope_q[i], 29000)) for i in range(32)]
        drope_line = VMobject(color=RED, stroke_width=3)
        drope_line.set_points_smoothly(drope_points)

        drope_legend = Text("DroPE", font_size=20, color=RED)
        drope_legend.move_to(axes.c2p(28, 25000))

        self.play(Create(drope_line), Write(drope_legend))

        # Highlight Layer 1
        layer1_dot = Dot(axes.c2p(1, min(drope_q[1], 29000)), color=YELLOW, radius=0.15)
        layer1_label = Text(f"Layer 1: {drope_q[1]:.0f}\n(149× RoPE)", font_size=16, color=YELLOW)
        layer1_label.next_to(layer1_dot, UP + RIGHT, buff=0.2)

        self.play(
            Create(layer1_dot),
            Write(layer1_label),
            Flash(layer1_dot, color=YELLOW, flash_radius=0.5)
        )

        # Explanation
        explanation = Text(
            "DroPE concentrates positional processing in Layer 1\nvia massive Q/K amplification",
            font_size=20,
            color=WHITE
        )
        explanation.to_edge(DOWN, buff=0.3)
        self.play(Write(explanation))

        self.wait(3)


class BOSAttentionSink(Scene):
    """
    Shows attention to BOS token across layers - the "sink" pattern.
    Both models show similar sink rates, but different functional dependence.
    """

    def construct(self):
        title = Text("Attention to BOS Token (Actual Measurements)", font_size=32)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Get BOS attention for each layer
        n_layers = 32

        rope_bos = []
        drope_bos = []

        for layer_idx in range(n_layers):
            rope_bos.append(np.mean(rope_data["layers"][str(layer_idx)]["bos_attention"][1:]))  # Exclude BOS itself
            drope_bos.append(np.mean(drope_data["layers"][str(layer_idx)]["bos_attention"][1:]))

        # Create axes
        axes = Axes(
            x_range=[0, 32, 4],
            y_range=[0, 1, 0.2],
            x_length=10,
            y_length=4,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": list(range(0, 33, 8))},
            y_axis_config={"numbers_to_include": [0, 0.5, 1.0]},
        )
        axes.center()

        x_label = Text("Layer", font_size=20).next_to(axes.x_axis, DOWN, buff=0.3)
        y_label = Text("Mean Attention to BOS", font_size=18).next_to(axes.y_axis, LEFT, buff=0.3).rotate(90*DEGREES)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # RoPE line
        rope_points = [axes.c2p(i, rope_bos[i]) for i in range(n_layers)]
        rope_line = VMobject(color=GREEN, stroke_width=3)
        rope_line.set_points_smoothly(rope_points)

        # DroPE line
        drope_points = [axes.c2p(i, drope_bos[i]) for i in range(n_layers)]
        drope_line = VMobject(color=RED, stroke_width=3)
        drope_line.set_points_smoothly(drope_points)

        rope_legend = Text("RoPE", font_size=20, color=GREEN).move_to(axes.c2p(28, 0.3))
        drope_legend = Text("DroPE", font_size=20, color=RED).move_to(axes.c2p(28, 0.15))

        self.play(Create(rope_line), Create(drope_line), Write(rope_legend), Write(drope_legend))

        # Key insight
        insight = Text(
            "Both models route ~70% attention to BOS\nBUT: Only RoPE depends on it (1249× PPL when ablated)",
            font_size=18,
            color=YELLOW
        )
        insight.to_edge(DOWN, buff=0.5)

        self.play(Write(insight))
        self.wait(3)


class FullComparison(Scene):
    """
    Complete narrative showing the key finding.
    """

    def construct(self):
        # Part 1: The question
        q1 = Text("If massive values come from RoPE...", font_size=30)
        q2 = Text("...and massive values are essential for context understanding...", font_size=30)
        q3 = Text("...how can DroPE (RoPE removed) still work?", font_size=30, color=YELLOW)

        q1.shift(UP * 1)
        q2.shift(UP * 0)
        q3.shift(DOWN * 1)

        self.play(Write(q1))
        self.wait(1)
        self.play(Write(q2))
        self.wait(1)
        self.play(Write(q3))
        self.wait(2)

        self.play(FadeOut(q1), FadeOut(q2), FadeOut(q3))

        # Part 2: The answer
        answer = Text("DroPE reorganizes its architecture", font_size=36, color=GREEN)
        self.play(Write(answer))
        self.wait(1)
        self.play(answer.animate.to_edge(UP, buff=0.5))

        # Show the key numbers
        stats = VGroup(
            Text("Layer 1 Q/K norms: 149× amplification", font_size=24),
            Text("Layer 1 attention contribution: 0.9% → 68.8%", font_size=24),
            Text("Massive value reliance: 82× less than RoPE", font_size=24),
            Text("Passkey under disruption: 0% degradation (vs 100% for RoPE)", font_size=24),
        ).arrange(DOWN, buff=0.4)
        stats.center()

        for stat in stats:
            self.play(Write(stat))
            self.wait(0.8)

        self.wait(2)

        # Conclusion
        self.play(FadeOut(stats))
        conclusion = Text(
            "DroPE learns alternative attention mechanisms\nthat don't rely on concentrated features",
            font_size=28,
            color=YELLOW
        )
        conclusion.center()
        self.play(Write(conclusion))
        self.wait(3)


if __name__ == "__main__":
    # To render: manim -pql scripts/attention_animation.py AttentionFlowComparison
    pass
