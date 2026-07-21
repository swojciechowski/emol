from itertools import zip_longest


def safe_latex(string):
    return string.replace('_', '\\_')


def latex_table(table, headers, width=None):
    lines = []

    width = f"{width}pt" if width else "\\linewidth"

    lines.append(f"\\begin{{tabularx}}{{{width}}}{{l{'C' * len(headers)}}}")
    lines.append("\\toprule")
    lines.append(' & ' + ' & '.join(map(safe_latex, headers)) + " \\\\")
    lines.append("\\midrule")

    for row in table:
        for a in zip_longest(*map(lambda _: _.split('\n'), row), fillvalue=''):
            lines.append(' & '.join(map(safe_latex, a)) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")

    return '\n'.join(lines)