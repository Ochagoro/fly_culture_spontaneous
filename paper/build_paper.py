"""Number citations in order of first appearance and build the paper HTML from its source.

Citations are written as [[key1,key2]] in the source. "ref. [[key]]" becomes "ref. N";
every other citation becomes a superscript with ranges compressed (1–3).
Usage: python build_paper.py [src dst]
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

REFS = {
    "wagenaar": "Wagenaar, D. A., Pine, J. &amp; Potter, S. M. An extremely rich repertoire of bursting patterns during the development of cortical cultures. <i>BMC Neurosci.</i> <b>7</b>, 11 (2006).",
    "eytan": "Eytan, D. &amp; Marom, S. Dynamics and effective topology underlying synchronization in networks of cortical neurons. <i>J. Neurosci.</i> <b>26</b>, 8465–8476 (2006).",
    "eckmann": "Eckmann, J.-P., Jacobi, S., Marom, S., Moses, E. &amp; Zbinden, C. Leader neurons in population bursts of 2D living neural networks. <i>New J. Phys.</i> <b>10</b>, 015011 (2008).",
    "orlandi": "Orlandi, J. G., Soriano, J., Alvarez-Lacalle, E., Teller, S. &amp; Casademunt, J. Noise focusing and the emergence of coherent activity in neuronal cultures. <i>Nat. Phys.</i> <b>9</b>, 582–590 (2013).",
    "lonardoni": "Lonardoni, D. et al. Recurrently connected and localized neuronal communities initiate coordinated spontaneous activity in neuronal networks. <i>PLoS Comput. Biol.</i> <b>13</b>, e1005672 (2017).",
    "okujeni": "Okujeni, S., Kandler, S. &amp; Egert, U. Mesoscale architecture shapes initiation and richness of spontaneous network activity. <i>J. Neurosci.</i> <b>37</b>, 3972–3987 (2017).",
    "pasquale": "Pasquale, V., Massobrio, P., Bologna, L. L., Chiappalone, M. &amp; Martinoia, S. Self-organization and neuronal avalanches in networks of dissociated cortical neurons. <i>Neuroscience</i> <b>153</b>, 1354–1369 (2008).",
    "tetzlaff": "Tetzlaff, C., Okujeni, S., Egert, U., Wörgötter, F. &amp; Butz, M. Self-organized criticality in developing neuronal networks. <i>PLoS Comput. Biol.</i> <b>6</b>, e1001013 (2010).",
    "beggs": "Beggs, J. M. &amp; Plenz, D. Neuronal avalanches in neocortical circuits. <i>J. Neurosci.</i> <b>23</b>, 11167–11177 (2003).",
    "wilting": "Wilting, J. &amp; Priesemann, V. Inferring collective dynamical states from widely unobserved systems. <i>Nat. Commun.</i> <b>9</b>, 2325 (2018).",
    "zierenberg": "Zierenberg, J., Wilting, J. &amp; Priesemann, V. Homeostatic plasticity and external input shape neural network dynamics. <i>Phys. Rev. X</i> <b>8</b>, 031018 (2018).",
    "yamamoto": "Yamamoto, H. et al. Impact of modular organization on dynamical richness in cortical networks. <i>Sci. Adv.</i> <b>4</b>, eaau4914 (2018).",
    "hinard": "Hinard, V. et al. Key electrophysiological, molecular, and metabolic signatures of sleep and wakefulness revealed in primary cortical cultures. <i>J. Neurosci.</i> <b>32</b>, 12506–12517 (2012).",
    "tsodyks": "Tsodyks, M. V. &amp; Markram, H. The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. <i>Proc. Natl Acad. Sci. USA</i> <b>94</b>, 719–723 (1997).",
    "nakamuta": "Nakamuta, A. et al. Self-organization of high-dimensional geometry of neural activity in culture. Preprint at bioRxiv https://doi.org/10.1101/2025.01.14.630383 (2025).",
    "dandi": "Nakamuta, A. et al. High-density MEA recordings of spontaneous activity in developing rat cortical cultures. DANDI Archive https://doi.org/10.48324/dandi.002020/0.261004.1410 (2026).",
    "turner": "Turner, M. H., Mann, K. &amp; Clandinin, T. R. The connectome predicts resting-state functional connectivity across the <i>Drosophila</i> brain. <i>Curr. Biol.</i> <b>31</b>, 2386–2394.e3 (2021).",
    "mann": "Mann, K., Gallen, C. L. &amp; Clandinin, T. R. Whole-brain calcium imaging reveals an intrinsic functional network in <i>Drosophila</i>. <i>Curr. Biol.</i> <b>27</b>, 2389–2396.e4 (2017).",
    "schaffer": "Schaffer, E. S. et al. The spatial and temporal structure of neural activity across the fly brain. <i>Nat. Commun.</i> <b>14</b>, 5572 (2023).",
    "dork": "Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. <i>Nature</i> <b>634</b>, 124–138 (2024).",
    "schlegel": "Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of <i>Drosophila</i>. <i>Nature</i> <b>634</b>, 139–152 (2024).",
    "shiu": "Shiu, P. K. et al. A <i>Drosophila</i> computational brain model reveals sensorimotor processing. <i>Nature</i> <b>634</b>, 210–219 (2024).",
    "eckstein": "Eckstein, N. et al. Neurotransmitter classification from electron microscopy images at synaptic sites in <i>Drosophila melanogaster</i>. <i>Cell</i> <b>187</b>, 2574–2594.e23 (2024).",
    "pospisil": "Pospisil, D. A. et al. The fly connectome reveals a path to the effectome. <i>Nature</i> <b>634</b>, 201–209 (2024).",
    "lin": "Lin, A. et al. Network statistics of the whole-brain connectome of <i>Drosophila</i>. <i>Nature</i> <b>634</b>, 153–165 (2024).",
    "li2026": "Li, Q., Ping, W., Zhang, K. &amp; Wang, C. Connectome-constrained modeling identifies neurons and synapses that sustain spontaneous activity in <i>Drosophila</i>. Preprint at bioRxiv https://doi.org/10.64898/2026.08.21.745055 (2026).",
    "schlegel2021": "Schlegel, P. et al. Information flow, cell types and stereotypy in a full olfactory connectome. <i>eLife</i> <b>10</b>, e66018 (2021).",
    "huang2019": "Huang, Y.-C. et al. A single-cell level and connectome-derived computational model of the <i>Drosophila</i> brain. <i>Front. Neuroinform.</i> <b>12</b>, 99 (2019).",
    "hasenhuetl": "Hasenhuetl, P. S. et al. A half-center oscillator regulates sleep pressure. <i>Nat. Neurosci.</i> https://doi.org/10.1038/s41593-026-02431-9 (2026).",
    "tononi": "Tononi, G. &amp; Cirelli, C. Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration. <i>Neuron</i> <b>81</b>, 12–34 (2014).",
    "bushey": "Bushey, D., Tononi, G. &amp; Cirelli, C. Sleep and synaptic homeostasis: structural evidence in <i>Drosophila</i>. <i>Science</i> <b>332</b>, 1576–1581 (2011).",
    "raccuglia": "Raccuglia, D. et al. Network-specific synchronization of electrical slow-wave oscillations regulates sleep drive in <i>Drosophila</i>. <i>Curr. Biol.</i> <b>29</b>, 3611–3621.e3 (2019).",
    "donlea": "Donlea, J. M. et al. Recurrent circuitry for balancing sleep need and sleep. <i>Neuron</i> <b>97</b>, 378–389.e4 (2018).",
    "shang": "Shang, Y., Claridge-Chang, A., Sjulson, L., Pypaert, M. &amp; Miesenböck, G. Excitatory local circuits and their implications for olfactory processing in the fly antennal lobe. <i>Cell</i> <b>128</b>, 601–612 (2007).",
    "olsen2007": "Olsen, S. R., Bhandawat, V. &amp; Wilson, R. I. Excitatory interactions between olfactory processing channels in the <i>Drosophila</i> antennal lobe. <i>Neuron</i> <b>54</b>, 89–103 (2007).",
    "huang2010": "Huang, J., Zhang, W., Qiao, W., Hu, A. &amp; Wang, Z. Functional connectivity and selective odor responses of excitatory local interneurons in <i>Drosophila</i> antennal lobe. <i>Neuron</i> <b>67</b>, 1021–1033 (2010).",
    "yaksi": "Yaksi, E. &amp; Wilson, R. I. Electrical coupling between olfactory glomeruli. <i>Neuron</i> <b>67</b>, 1034–1047 (2010).",
    "seki": "Seki, Y., Rybak, J., Wicher, D., Sachse, S. &amp; Hansson, B. S. Physiological and morphological characterization of local interneurons in the <i>Drosophila</i> antennal lobe. <i>J. Neurophysiol.</i> <b>104</b>, 1007–1019 (2010).",
    "chou": "Chou, Y.-H. et al. Diversity and wiring variability of olfactory local interneurons in the <i>Drosophila</i> antennal lobe. <i>Nat. Neurosci.</i> <b>13</b>, 439–449 (2010).",
    "kazama": "Kazama, H. &amp; Wilson, R. I. Origins of correlated activity in an olfactory circuit. <i>Nat. Neurosci.</i> <b>12</b>, 1136–1144 (2009).",
    "wilsonlaurent": "Wilson, R. I. &amp; Laurent, G. Role of GABAergic inhibition in shaping odor-evoked spatiotemporal patterns in the <i>Drosophila</i> antennal lobe. <i>J. Neurosci.</i> <b>25</b>, 9069–9079 (2005).",
    "olsen2008": "Olsen, S. R. &amp; Wilson, R. I. Lateral presynaptic inhibition mediates gain control in an olfactory circuit. <i>Nature</i> <b>452</b>, 956–960 (2008).",
    "galan": "Galán, R. F., Weidert, M., Menzel, R., Herz, A. V. M. &amp; Galizia, C. G. Sensory memory for odors is encoded in spontaneous correlated activity between olfactory glomeruli. <i>Neural Comput.</i> <b>18</b>, 10–25 (2006).",
    "piredda": "Piredda, S. &amp; Gale, K. A crucial epileptogenic site in the deep prepiriform cortex. <i>Nature</i> <b>317</b>, 623–625 (1985).",
    "loscher": "Löscher, W. &amp; Ebert, U. The role of the piriform cortex in kindling. <i>Prog. Neurobiol.</i> <b>50</b>, 427–481 (1996).",
    "galovic": "Galovic, M. et al. Association of piriform cortex resection with surgical outcomes in patients with temporal lobe epilepsy. <i>JAMA Neurol.</i> <b>76</b>, 690–700 (2019).",
    "marin": "Marin, E. C. et al. Connectomics analysis reveals first-, second-, and third-order thermosensory and hygrosensory neurons in the adult <i>Drosophila</i> brain. <i>Curr. Biol.</i> <b>30</b>, 3167–3182.e4 (2020).",
    "yadlapalli": "Yadlapalli, S. et al. Circadian clock neurons constantly monitor environmental temperature to set sleep timing. <i>Nature</i> <b>555</b>, 98–102 (2018).",
    "repo": "Shiu, P. K. Drosophila_brain_model. GitHub https://github.com/philshiu/Drosophila_brain_model (2024).",
    "brian": "Stimberg, M., Brette, R. &amp; Goodman, D. F. M. Brian 2, an intuitive and efficient neural simulator. <i>eLife</i> <b>8</b>, e47314 (2019).",
    "maxone": "Müller, J. et al. High-resolution CMOS MEA platform to study neurons at subcellular, cellular, and network levels. <i>Lab Chip</i> <b>15</b>, 2767–2780 (2015).",
}


def compress(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def build(src="paper_en_src.html", dst="paper_en.html"):
    s = (HERE / src).read_text()
    order = {}
    for m in re.finditer(r"\[\[([a-z0-9_,]+)\]\]", s):
        for k in m.group(1).split(","):
            if k not in REFS:
                raise KeyError(k)
            order.setdefault(k, len(order) + 1)

    def repl(m):
        nums = [order[k] for k in m.group(2).split(",")]
        if m.group(1):
            return m.group(1) + compress(nums)
        return f"<sup>{compress(nums)}</sup>"

    s = re.sub(r"(ref\. |refs\. |文献 )?\[\[([a-z0-9_,]+)\]\]", repl, s)
    refs = "<ol class=\"refs\">\n" + "\n".join(f"<li>{REFS[k]}</li>" for k in sorted(order, key=order.get)) + "\n</ol>"
    s = s.replace("{{REFERENCES}}", refs)

    def words(html):
        t = re.sub(r"<figure>.*?</figure>", " ", html, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"&[a-z]+;", " ", t)
        return len(t.split())
    if "{{WORDCOUNT}}" in s:
        a0 = s.index('<p class="summary">')
        summ = words(s[a0:s.index("</p>", a0)])
        main = words(s[s.index("<h2>Main</h2>"):s.index("<h2>Methods</h2>")])
        meth = words(s[s.index("<h2>Methods</h2>"):s.index("<h2>References</h2>")])
        s = s.replace("{{WORDCOUNT}}", f"Article (submission format) · Summary {summ} words · Main text {main:,} words · "
                      f"Methods {meth:,} words · 4 figures · 4 Extended Data figures · {len(order)} references")
    (HERE / dst).write_text(s)
    unused = [k for k in REFS if k not in order]
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", s)
    print("refs", len(order), "unused", unused, "placeholders left", left)


if __name__ == "__main__":
    build(*sys.argv[1:3])
