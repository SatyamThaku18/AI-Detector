import matplotlib
matplotlib.use('Agg')  # 🔥 IMPORTANT FIX

import matplotlib.pyplot as plt

def generate_graph(fake_scores, real_scores):
    total_fake = len(fake_scores)
    total_real = len(real_scores)

    if total_fake == 0 and total_real == 0:
        total_fake = 1

    # Bar chart
    plt.figure()
    plt.bar(["Fake", "Real"], [total_fake, total_real])
    plt.title("Frame Distribution")
    plt.savefig("static/bar_graph.png")
    plt.close()

    # Pie chart
    plt.figure()
    plt.pie(
        [total_fake, total_real],
        labels=["Fake", "Real"],
        autopct='%1.1f%%'
    )
    plt.title("Fake vs Real")
    plt.savefig("static/pie_chart.png")
    plt.close()

    # Line graph
    plt.figure()
    if fake_scores:
        plt.plot(fake_scores, label="Fake", marker='o')
    if real_scores:
        plt.plot(real_scores, label="Real", marker='o')

    plt.legend()
    plt.title("Confidence Trend")
    plt.savefig("static/line_graph.png")
    plt.close()
