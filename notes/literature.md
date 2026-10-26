# Literature notes: antennal-lobe excitatory LNs as spontaneous "ignition" hub (2026-10-27)

Sources checked: Crossref API (metadata for every Part B entry), PubMed abstracts (E-utilities), Europe PMC / PMC full text where open (Kazama & Wilson 2009; Yaksi & Wilson 2010; Shang et al. 2007; Olsen & Wilson 2008; Schlegel et al. 2021). Publisher full texts (Cell Press, J Neurosci, J Neurophysiol, Nature) were mostly blocked (HTTP 403), so some numbers below are **abstract-level only**, and these are flagged.

Tags: **[SUPPORTS]** / **[AGAINST]** / **[NEUTRAL]** the model prediction that cholinergic AL LNs (lLN1_bc, lLN2X12) form a self-exciting recurrent hub that ignites spontaneous bursts when gain is raised.

---

## Part A: Findings

### A1. Spontaneous activity of AL local neurons, especially excitatory (cholinergic) LNs

- **[SUPPORTS: identity and number] Shang et al. 2007 (Cell 128:601).** This paper first described multiglomerular **cholinergic LNs (eLNs)**. ChAT immunostaining found ~15 cholinergic LNs per antennal lobe, and the GAL4 lines labelled fewer (10–12). In KL107-GAL4 and krasavietz-GAL4, 71% and 63% of GFP+ cells were ChAT+. eLNs respond broadly to odours and send diffuse, glomerulus-nonspecific excitation to PNs, which the authors liken to stochastic resonance. Full text: GABA blockade (picrotoxin 250 µM + CGP54626 50 µM) caused only a "modest generalized increase in PN response amplitudes" (4.02 → 4.73 % ΔF/F). The authors also found no evidence for tonic inhibition of PNs. The paper gives no spontaneous rates in Hz.
- **[SUPPORTS: lateral excitation exists] Olsen, Bhandawat & Wilson 2007 (Neuron 54:89).** PNs whose own ORNs are silenced still get substantial lateral excitatory input from other glomeruli. These lateral connections are stereotyped across flies.
- **[SUPPORTS: recurrent eLN–eLN and eLN↔PN loops] Huang et al. 2010 (Neuron 67:1021).** Paired recordings showed **reciprocal** krasavietz eLN↔PN excitatory connections, carried by **both dendrodendritic cholinergic synapses and gap junctions**. They also found reciprocal **eLN↔eLN** connections, while eLN↔iLN connections were rare. eLNs get monosynaptic ORN input and show odour-specific temporal spiking patterns (abstract-level; no spontaneous rate retrieved).
- **[MIXED: mechanism] Yaksi & Wilson 2010 (Neuron 67:1034).** eLN→PN synapses are **electrical**. They pass both depolarization and hyperpolarization, persist when chemical transmission is blocked, and are abolished by the gap-junction mutant *shakB2*. The authors conclude that "eLNs do not release acetylcholine onto PNs". eLN→**iLN** synapses are "largely cholinergic", blocked by Cd2+ or mecamylamine. So eLNs drive both direct excitation and indirect inhibition of PNs. In vivo, eLNs "were always barraged by spontaneous IPSPs", meaning they sit under ongoing inhibition. Antennae were removed "to minimize spontaneous activity". *shakB2* flies are noted to have increased seizure susceptibility. **Caveat for the model:** FlyWire has no gap junctions, so the model's eLN→PN drive is cholinergic-chemical, which disagrees with this mechanism.
- **[AGAINST: identity of krasavietz LNs] Seki et al. 2010 (J Neurophysiol 104:1007).** This study characterised four LN classes. One class has **burst-firing** properties and the others fire tonically. It also found that krasavietz-GAL4 LNs, previously reported as cholinergic, "were mostly GABAergic" (abstract-level). This means part of the "eLN" physiology literature may have sampled mixed populations.
- **[NEUTRAL] Chou et al. 2010 (Nat Neurosci 13:439).** Survey of >1,500 LNs. Transmitter profiles are diverse (cholinergic, GABAergic, glutamatergic). Morphological classes are coarsely stereotyped but wiring and physiology vary widely between individuals, and some classes differ physiologically. I could not retrieve the full text (PMC behind captcha), so **class-specific spontaneous rates are not recorded here**.
- **[NEUTRAL: iLNs] Nagel & Wilson 2016 (J Neurosci 36:4325).** iLNs show a continuum of ON/OFF responses and preferred timescales, set by the E/I synaptic balance (ON vs OFF) and by intrinsic properties (timescale). This is odour-driven dynamics only; no spontaneous-burst data in the abstract.
- **[AGAINST: in vivo spontaneous activity is ORN-driven, not AL-intrinsic] Kazama & Wilson 2009 (Nat Neurosci 12:1136, full text).** PNs fire spontaneously at "typically 1–5 spikes/sec". Spontaneous spikes are correlated only between **homotypic** PNs (same ORN type), and "uncorrelated in heterotypic pairs". Spontaneous EPSCs arrive at ~75 Hz in DM4 PNs, which equals ORN rate × ORN number (3.44 Hz × 17.4). Supplementary Fig. 3 states that spontaneous membrane-potential fluctuations "are mainly driven by direct ORN input, not central input". Taken together, this argues against spontaneous AL-wide synchrony from eLN recurrence under normal gain. The Discussion notes "spontaneous non-oscillatory synchronous bursting in different glomeruli ... on slower time scales (~seconds)" in the honeybee AL (Galán et al. 2006, Neural Comput 18:10), which **[SUPPORTS]** slow, coordinated spontaneous AL fluctuations of the kind Turner et al. 2021 saw.
- **Summary of spontaneous rates retrieved.** PNs fire at 1–5 Hz (Kazama & Wilson 2009). ORNs fire at roughly 1–10 Hz depending on type (VM7 ~10 Hz, Olsen & Wilson 2008; DM4 3.4 Hz). **eLN and iLN spontaneous rates in Hz: UNVERIFIED.** They are not reported in the abstracts or open full texts I could read; Huang 2010, Seki 2010 and Chou 2010 likely contain them but were paywalled.
- **eLN→PN coupling: answer.** It is **electrical** (Yaksi & Wilson 2010), or **electrical plus cholinergic** (Huang et al. 2010). The eLN→iLN coupling is chemical and cholinergic (Yaksi & Wilson 2010).

### A2. GABA blockade, epileptiform activity in insect ALs, fly seizure onset

- **[NEUTRAL] Wilson & Laurent 2005 (J Neurosci 25:9069).** Many axonless LNs are GABAergic. GABA hyperpolarizes PNs through a picrotoxin-sensitive GABA_A conductance and a CGP54626-sensitive GABA_B conductance. GABA_A shapes the early odour response and GABA_B mediates slower inhibition. The abstract gives no report of spontaneous bursts under blockade (full text blocked).
- **[AGAINST, weakly] Olsen & Wilson 2008 (Nature 452:956).** Removing interglomerular input broadens PN tuning: "inter-glomerular inhibition dominates over inter-glomerular excitation". Inhibition scales with total AL input and is largely presynaptic (GABA_A + GABA_B on ORN terminals). Picrotoxin + CGP54626 produced "strong disinhibition that resembled the effect of removing the antennae" for odour responses. **No spontaneous runaway or bursting is reported.**
- **[NEUTRAL] Yaksi & Wilson 2010.** The disinhibitory effect of picrotoxin + CGP54626 on PNs is smaller in *shakB2*. In other words, part of what GABA blockade unmasks is eLN-borne excitation, which fits the eLN network being held in check by inhibition.
- **Insect ALs (locust/moth/bee).** In locust, picrotoxin abolishes odour-evoked 20–30 Hz PN synchrony and LFP oscillations but spares slow patterning (MacLeod & Laurent 1996, Science 274:976). In *Manduca*, picrotoxin/bicuculline block PN IPSPs, and GABA suppresses spontaneous spiking (Waldrop, Christensen & Hildebrand 1987, J Comp Physiol A; details not re-verified). **I found no report of spontaneous epileptiform or synchronous bursts in any insect AL after GABA blockade.** This absence is unconfirmed rather than demonstrated: only a quick search was done.
- **Fly seizure localisation.** I found no study that maps seizure **onset** to a particular adult brain region. Bang-sensitive seizure assays read out flight muscle (DLM) or giant-fibre activity, and calcium imaging has been done in larval motoneurons (Streit et al. 2016, PLoS One 11:e0148461) or in ex vivo whole brains without regional onset analysis. **Huang et al. 2019 (Front Neuroinform 12:99)** is real: a FlyCircuit-based whole-brain LIF model (20,089 neurons, 1,044,020 synapses). The reconstructed network was stabilised by short-term depression, whereas the degree-preserving shuffled network "exhibited seizure-like firing activity". This is a useful precedent: connectome structure plus STD prevents runaway activity.

### A3. Mammalian analogy: olfactory/piriform cortex as most seizure-prone

- **[SUPPORTS, by analogy] Piredda & Gale 1985 (Nature 317:623).** Unilateral bicuculline microinjection into a discrete site of the deep prepiriform cortex ("area tempestas") triggers generalized limbic motor seizures. It is the most sensitive chemoconvulsant trigger zone in the rat forebrain (content from memory of the classic result; abstract not indexed).
- **Löscher & Ebert 1996 (Prog Neurobiol 50:427).** The piriform cortex has the lowest kindling threshold and acts as a critical node for kindled-seizure generalization. Its extensive recurrent excitatory associational connections underlie this.
- **Vismer et al. 2015 (Front Neural Circuits 9:27).** Review: piriform, perirhinal and entorhinal cortex in seizure generation, with area tempestas as a trigger zone.
- **Galovic et al. 2019 (JAMA Neurol 76:690).** In TLE surgery, seizure-free patients had a larger fraction of piriform cortex resected (median 83% vs 52%, P<.001). The finding was validated in two external cohorts, and it supports a key role for piriform cortex in human seizure generation.
- The take-home analogy: the first-order olfactory cortex, with dense recurrent excitation, is the brain's most epileptogenic region. That parallels the model's finding that the AL's cholinergic recurrent loop is the fly brain's most excitable module.

### A4. Thermo/hygrosensory AL pathways → clock neurons → sleep

- **Frank et al. 2017 (Curr Biol 27:2381).** Ir68a "humid" neurons complete the hygrosensory triad. The posterior AL holds adjacent hot, cold, dry and humid glomeruli, and at least one PN type integrates temperature with dry-air input.
- **Marin et al. 2020 (Curr Biol 30:3167).** EM connectomics of thermo- and hygrosensory VP glomeruli and VP PNs, which feed the lateral accessory calyx (lACA). "DN1a neurons link thermosensory PNs in the lACA to the circadian clock."
- **Alpert et al. 2020 (Curr Biol 30:2275).** Cold TPN-IIs fire persistently and scale with absolute temperature below ~25 °C. They sit in a "thermometer" circuit that gives persistent inhibition to centres regulating sleep and activity.
- **Alpert et al. 2022 (Curr Biol 32:4079).** Hot thermometer: internal AC neurons and antennal thermosensors (via a new PN class, TPN-IV) drive LPN clock neurons. Silencing LPNs blocks heat-induced restructuring of the siesta.
- **Yadlapalli et al. 2018 (Nature 555:98).** Sleep-promoting DN1p clock neurons are inhibited by heating and excited by cooling. They depend on peripheral thermoreceptors in the aristae and chordotonal organs, and they set sleep timing under naturalistic temperature cycles.
- **Jin et al. 2021 (Curr Biol 31:2075).** Warmth activates internal AC thermosensors, which release ACh onto a DN1p subset. That subset promotes wake through CNMa signalling onto Dh44 PI neurons. This is a non-AL route.
- **No source located** for the "humidity → DN1 → sleep" link. Marin 2020 gives the anatomical VP PN → DN1a link for thermosensation only.

---

## Part B: Full references (Nature style)

Verified against Crossref (and DataCite for DANDI) unless marked otherwise. Consortium author lists are abbreviated as "et al."

**Cultures / criticality**
1. Eytan, D. & Marom, S. Dynamics and effective topology underlying synchronization in networks of cortical neurons. *J. Neurosci.* **26**, 8465–8476 (2006). doi:10.1523/JNEUROSCI.1627-06.2006
2. Eckmann, J.-P., Jacobi, S., Marom, S., Moses, E. & Zbinden, C. Leader neurons in population bursts of 2D living neural networks. *New J. Phys.* **10**, 015011 (2008). doi:10.1088/1367-2630/10/1/015011
3. Orlandi, J. G., Soriano, J., Alvarez-Lacalle, E., Teller, S. & Casademunt, J. Noise focusing and the emergence of coherent activity in neuronal cultures. *Nat. Phys.* **9**, 582–590 (2013). doi:10.1038/nphys2686
4. Lonardoni, D. et al. Recurrently connected and localized neuronal communities initiate coordinated spontaneous activity in neuronal networks. *PLoS Comput. Biol.* **13**, e1005672 (2017). doi:10.1371/journal.pcbi.1005672
5. Okujeni, S., Kandler, S. & Egert, U. Mesoscale architecture shapes initiation and richness of spontaneous network activity. *J. Neurosci.* **37**, 3972–3987 (2017). doi:10.1523/JNEUROSCI.2552-16.2017
6. Wagenaar, D. A., Pine, J. & Potter, S. M. An extremely rich repertoire of bursting patterns during the development of cortical cultures. *BMC Neurosci.* **7**, 11 (2006). doi:10.1186/1471-2202-7-11
7. Pasquale, V., Massobrio, P., Bologna, L. L., Chiappalone, M. & Martinoia, S. Self-organization and neuronal avalanches in networks of dissociated cortical neurons. *Neuroscience* **153**, 1354–1369 (2008). doi:10.1016/j.neuroscience.2008.03.050
8. Tetzlaff, C., Okujeni, S., Egert, U., Wörgötter, F. & Butz, M. Self-organized criticality in developing neuronal networks. *PLoS Comput. Biol.* **6**, e1001013 (2010). doi:10.1371/journal.pcbi.1001013
9. Beggs, J. M. & Plenz, D. Neuronal avalanches in neocortical circuits. *J. Neurosci.* **23**, 11167–11177 (2003). doi:10.1523/JNEUROSCI.23-35-11167.2003
10. Wilting, J. & Priesemann, V. Inferring collective dynamical states from widely unobserved systems. *Nat. Commun.* **9**, 2325 (2018). doi:10.1038/s41467-018-04725-4
11. Zierenberg, J., Wilting, J. & Priesemann, V. Homeostatic plasticity and external input shape neural network dynamics. *Phys. Rev. X* **8**, 031018 (2018). doi:10.1103/PhysRevX.8.031018
12. Yamamoto, H. et al. Impact of modular organization on dynamical richness in cortical networks. *Sci. Adv.* **4**, eaau4914 (2018). doi:10.1126/sciadv.aau4914
13. Hinard, V. et al. Key electrophysiological, molecular, and metabolic signatures of sleep and wakefulness revealed in primary cortical cultures. *J. Neurosci.* **32**, 12506–12517 (2012). doi:10.1523/JNEUROSCI.2306-12.2012
14. Kinouchi, O. & Copelli, M. Optimal dynamical range of excitable networks at criticality. *Nat. Phys.* **2**, 348–351 (2006). doi:10.1038/nphys289
15. Tsodyks, M. V. & Markram, H. The neural code between neocortical pyramidal neurons depends on neurotransmitter release probability. *Proc. Natl Acad. Sci. USA* **94**, 719–723 (1997). doi:10.1073/pnas.94.2.719
16. Nakamuta, A., Akita, D., Zhang, H., Kawahara, Y., Takahashi, H. & Teramae, J. Self-organization of high-dimensional geometry of neural activity in culture. Preprint at *bioRxiv* https://doi.org/10.1101/2025.01.14.630383 (2025).
17. Nakamuta, A., Akita, D., Zhang, H., Kawahara, Y., Takahashi, H. & Teramae, J. High-density MEA recordings of spontaneous activity in developing rat cortical cultures. *DANDI Archive* (version 0.261004.1410) https://doi.org/10.48324/dandi.002020/0.261004.1410 (2026). (DataCite-verified: title, creators, version, year 2026.)

**Fly whole-brain imaging / connectome / models**
18. Turner, M. H., Mann, K. & Clandinin, T. R. The connectome predicts resting-state functional connectivity across the *Drosophila* brain. *Curr. Biol.* **31**, 2386–2394.e3 (2021). doi:10.1016/j.cub.2021.03.004
19. Mann, K., Gallen, C. L. & Clandinin, T. R. Whole-brain calcium imaging reveals an intrinsic functional network in *Drosophila*. *Curr. Biol.* **27**, 2389–2396.e4 (2017). doi:10.1016/j.cub.2017.06.076
20. Schaffer, E. S. et al. The spatial and temporal structure of neural activity across the fly brain. *Nat. Commun.* **14**, 5572 (2023). doi:10.1038/s41467-023-41261-2
21. Dorkenwald, S. et al. Neuronal wiring diagram of an adult brain. *Nature* **634**, 124–138 (2024). doi:10.1038/s41586-024-07558-y
22. Schlegel, P. et al. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* **634**, 139–152 (2024). doi:10.1038/s41586-024-07686-5
23. Shiu, P. K. et al. A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* **634**, 210–219 (2024). doi:10.1038/s41586-024-07763-9
24. Eckstein, N. et al. Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* **187**, 2574–2594.e23 (2024). doi:10.1016/j.cell.2024.03.016
25. Pospisil, D. A. et al. The fly connectome reveals a path to the effectome. *Nature* **634**, 201–209 (2024). doi:10.1038/s41586-024-07982-0
26. Lin, A. et al. Network statistics of the whole-brain connectome of *Drosophila*. *Nature* **634**, 153–165 (2024). doi:10.1038/s41586-024-07968-y
27. Li, Q., Ping, W., Zhang, K. & Wang, C. Connectome-constrained modeling identifies neurons and synapses that sustain spontaneous activity in *Drosophila*. Preprint at *bioRxiv* https://doi.org/10.64898/2026.08.21.745055 (2026). (DOI resolves in Crossref, posted 2026-08-25; that the model is fit to Turner et al. data is NOT checked against the text.)
28. Schlegel, P. et al. Information flow, cell types and stereotypy in a full olfactory connectome. *eLife* **10**, e66018 (2021). doi:10.7554/eLife.66018
29. Huang, Y.-C. et al. A single-cell level and connectome-derived computational model of the *Drosophila* brain. *Front. Neuroinform.* **12**, 99 (2019). doi:10.3389/fninf.2018.00099. (Real: FlyCircuit-based Flysim LIF model; STD stabilises the real network, shuffled network shows seizure-like activity. Note that the volume is 12 although it was published in Jan 2019.)

**Sleep**
30. Hasenhuetl, P. S., Sarnataro, R., Vrontou, E., Rorsman, H. O., Talbot, C. B., Brain, R. & Miesenböck, G. A half-center oscillator regulates sleep pressure. *Nat. Neurosci.* https://doi.org/10.1038/s41593-026-02431-9 (2026). (Published online 2026-10-15; volume and pages not yet assigned. The bioRxiv title was "A half-center oscillator encodes sleep pressure", doi:10.1101/2024.02.23.581780.)
31. Tononi, G. & Cirelli, C. Sleep and the price of plasticity: from synaptic and cellular homeostasis to memory consolidation and integration. *Neuron* **81**, 12–34 (2014). doi:10.1016/j.neuron.2013.12.025
32. Bushey, D., Tononi, G. & Cirelli, C. Sleep and synaptic homeostasis: structural evidence in *Drosophila*. *Science* **332**, 1576–1581 (2011). doi:10.1126/science.1202839
33. Gilestro, G. F., Tononi, G. & Cirelli, C. Widespread changes in synaptic markers as a function of sleep and wakefulness in *Drosophila*. *Science* **324**, 109–112 (2009). doi:10.1126/science.1166673
34. Raccuglia, D. et al. Network-specific synchronization of electrical slow-wave oscillations regulates sleep drive in *Drosophila*. *Curr. Biol.* **29**, 3611–3621.e3 (2019). doi:10.1016/j.cub.2019.08.070. (**CORRECTION:** the journal is *Current Biology*, not *Neuron*.)
35. Donlea, J. M. et al. Recurrent circuitry for balancing sleep need and sleep. *Neuron* **97**, 378–389.e4 (2018). doi:10.1016/j.neuron.2017.12.016. (Full title has "...sleep need and sleep".)

**Antennal lobe physiology (Part A)**
36. Shang, Y., Claridge-Chang, A., Sjulson, L., Pypaert, M. & Miesenböck, G. Excitatory local circuits and their implications for olfactory processing in the fly antennal lobe. *Cell* **128**, 601–612 (2007). doi:10.1016/j.cell.2006.12.034
37. Olsen, S. R., Bhandawat, V. & Wilson, R. I. Excitatory interactions between olfactory processing channels in the *Drosophila* antennal lobe. *Neuron* **54**, 89–103 (2007). doi:10.1016/j.neuron.2007.03.010
38. Huang, J., Zhang, W., Qiao, W., Hu, A. & Wang, Z. Functional connectivity and selective odor responses of excitatory local interneurons in *Drosophila* antennal lobe. *Neuron* **67**, 1021–1033 (2010). doi:10.1016/j.neuron.2010.08.025
39. Yaksi, E. & Wilson, R. I. Electrical coupling between olfactory glomeruli. *Neuron* **67**, 1034–1047 (2010). doi:10.1016/j.neuron.2010.08.041
40. Seki, Y., Rybak, J., Wicher, D., Sachse, S. & Hansson, B. S. Physiological and morphological characterization of local interneurons in the *Drosophila* antennal lobe. *J. Neurophysiol.* **104**, 1007–1019 (2010). doi:10.1152/jn.00249.2010
41. Chou, Y.-H. et al. Diversity and wiring variability of olfactory local interneurons in the *Drosophila* antennal lobe. *Nat. Neurosci.* **13**, 439–449 (2010). doi:10.1038/nn.2489
42. Nagel, K. I. & Wilson, R. I. Mechanisms underlying population response dynamics in inhibitory interneurons of the *Drosophila* antennal lobe. *J. Neurosci.* **36**, 4325–4338 (2016). doi:10.1523/JNEUROSCI.3887-15.2016
43. Kazama, H. & Wilson, R. I. Origins of correlated activity in an olfactory circuit. *Nat. Neurosci.* **12**, 1136–1144 (2009). doi:10.1038/nn.2376
44. Wilson, R. I. & Laurent, G. Role of GABAergic inhibition in shaping odor-evoked spatiotemporal patterns in the *Drosophila* antennal lobe. *J. Neurosci.* **25**, 9069–9079 (2005). doi:10.1523/JNEUROSCI.2070-05.2005
45. Olsen, S. R. & Wilson, R. I. Lateral presynaptic inhibition mediates gain control in an olfactory circuit. *Nature* **452**, 956–960 (2008). doi:10.1038/nature06864
46. Tobin, W. F., Wilson, R. I. & Lee, W.-C. A. Wiring variations that enable and constrain neural computation in a sensory microcircuit. *eLife* **6**, e24838 (2017). doi:10.7554/eLife.24838
47. Galán, R. F., Weidert, M., Menzel, R., Herz, A. V. M. & Galizia, C. G. Sensory memory for odors is encoded in spontaneous correlated activity between olfactory glomeruli. *Neural Comput.* **18**, 10–25 (2006). doi:10.1162/089976606774841558
48. MacLeod, K. & Laurent, G. Distinct mechanisms for synchronization and temporal patterning of odor-encoding neural assemblies. *Science* **274**, 976–979 (1996). doi:10.1126/science.274.5289.976
49. Streit, A. K., Fan, Y. N., Masullo, L. & Baines, R. A. Calcium imaging of neuronal activity in *Drosophila* can identify anticonvulsive compounds. *PLoS ONE* **11**, e0148461 (2016). doi:10.1371/journal.pone.0148461

**Piriform / area tempestas**
50. Piredda, S. & Gale, K. A crucial epileptogenic site in the deep prepiriform cortex. *Nature* **317**, 623–625 (1985). doi:10.1038/317623a0
51. Löscher, W. & Ebert, U. The role of the piriform cortex in kindling. *Prog. Neurobiol.* **50**, 427–481 (1996). doi:10.1016/S0301-0082(96)00036-6
52. Vismer, M. S., Forcelli, P. A., Skopin, M. D., Gale, K. & Koubeissi, M. Z. The piriform, perirhinal, and entorhinal cortex in seizure generation. *Front. Neural Circuits* **9**, 27 (2015). doi:10.3389/fncir.2015.00027
53. Galovic, M. et al. Association of piriform cortex resection with surgical outcomes in patients with temporal lobe epilepsy. *JAMA Neurol.* **76**, 690–700 (2019). doi:10.1001/jamaneurol.2019.0204

**Thermo/hygrosensation and sleep**
54. Yadlapalli, S. et al. Circadian clock neurons constantly monitor environmental temperature to set sleep timing. *Nature* **555**, 98–102 (2018). doi:10.1038/nature25740
55. Frank, D. D. et al. Early integration of temperature and humidity stimuli in the *Drosophila* brain. *Curr. Biol.* **27**, 2381–2388.e4 (2017). doi:10.1016/j.cub.2017.06.077
56. Marin, E. C. et al. Connectomics analysis reveals first-, second-, and third-order thermosensory and hygrosensory neurons in the adult *Drosophila* brain. *Curr. Biol.* **30**, 3167–3182.e4 (2020). doi:10.1016/j.cub.2020.06.028
57. Alpert, M. H. et al. A circuit encoding absolute cold temperature in *Drosophila*. *Curr. Biol.* **30**, 2275–2288.e5 (2020). doi:10.1016/j.cub.2020.04.038
58. Alpert, M. H., Gil, H., Para, A. & Gallio, M. A thermometer circuit for hot temperature adjusts *Drosophila* behavior to persistent heat. *Curr. Biol.* **32**, 4079–4087.e4 (2022). doi:10.1016/j.cub.2022.07.060
59. Jin, X. et al. A subset of DN1p neurons integrates thermosensory inputs to promote wakefulness via CNMa signaling. *Curr. Biol.* **31**, 2075–2087.e6 (2021). doi:10.1016/j.cub.2021.02.048

**UNVERIFIED / not checked**
- Waldrop, B., Christensen, T. A. & Hildebrand, J. G. (1987) *J. Comp. Physiol. A*: moth AL GABA/picrotoxin. Cited from search snippet only; volume and pages are not checked.
- Spontaneous firing rates (Hz) of eLNs and iLNs: not retrieved (see A1).
- Whether FlyWire lLN1_bc / lLN2X12 correspond to Shang's ChAT+ LNs or krasavietz eLNs: not confirmed. Schlegel et al. 2021 full text has no "lLN1" or "eLN" mapping statement.
