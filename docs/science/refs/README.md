# Reference full texts — what is here, and what is not

Backing material for [`../normal-modes.md`](?doc=science/normal-modes.md).
Every entry's bibliographic record was checked against Crossref and the
publisher on **2026-09-21**; the result of each check is recorded beside the
entry in [`../references.bib`](?doc=science/references.bib).

Each article is stored as the PDF **and** as a `.txt` rendering, so the prose
can be grepped and quoted without opening a reader. The `.txt` files are
machine extractions (Ghostscript `txtwrite`, or JATS XML for `Vester2024`);
equations and tables do not survive that process, so **quote from the PDF, not
from the text rendering**.

> ⚠ **Redistribution.** `Vester2024` is CC-BY and may be shared freely. The
> other three are publisher copies obtained through an institutional
> subscription — fine to hold for personal and group use, not to redistribute.
> If this repository is ever made public, add `docs/science/refs/*.pdf` to
> `.gitignore` and keep the `.txt` renderings out with them; `README.md` and
> `references.bib` carry everything needed to find the sources again.

## Held here

| file | citation | why it is here |
|---|---|---|
| `Vester2024-partial-hessian-qmmm.*` (xml + txt) | Vester & Olsen, *J. Chem. Theory Comput.* **20**(21), 9533–9546 (2024) · CC-BY 4.0 | the published argument **against** the naive fix; § 3.1 answers it |
| `Besley2008-phva-si100.*` (pdf + txt) | Besley & Bryan, *J. Phys. Chem. C* **112**(11), 4308–4314 (2008) | PHVA in practice; states which Hessian is sliced and the infinite-mass meaning |
| `Tao2021-revised-gsva.*` (pdf + txt) | Tao, Zou, Nanayakkara, Freindorf & Kraka, *Theor. Chem. Acc.* **140**(3), 31 (2021) | places PHVA among MBH / VSA / GSVA and states the limitation they share |
| `Ghysels2008-mbh-simbioma-abstract.*` (pdf + txt) | Ghysels, Van Speybroeck, Van Neck, Brooks & Waroquier — conference abstract, SimBioMa 2008 | one corroborating sentence: three zero eigenvalues, not six, at a partially optimised geometry |

`Vester2024` was retrieved from
`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11562069/fullTextXML`
(© the authors, CC-BY 4.0). The other three were supplied by the user from
institutional access on 2026-09-21.

## Not here — paywalled, no open copy exists

Checked against Unpaywall on 2026-09-21; all returned `is_oa: false`.

| key | citation | DOI |
|---|---|---|
| `Eckart1935` | Eckart, *Phys. Rev.* **47**(7), 552–558 (1935) | [10.1103/PhysRev.47.552](https://doi.org/10.1103/PhysRev.47.552) |
| `Head1997` | Head, *Int. J. Quantum Chem.* **65**(5), 827–838 (1997) | [10.1002/(SICI)1097-461X(1997)65:5<827::AID-QUA47>3.0.CO;2-U](https://doi.org/10.1002/%28SICI%291097-461X%281997%2965:5%3C827::AID-QUA47%3E3.0.CO;2-U) |
| `LiJensen2002` | Li & Jensen, *Theor. Chem. Acc.* **107**(4), 211–219 (2002) | [10.1007/s00214-001-0317-7](https://doi.org/10.1007/s00214-001-0317-7) |
| `Ghysels2007` | Ghysels *et al.*, *J. Chem. Phys.* **126**(22), 224102 (2007) | [10.1063/1.2737444](https://doi.org/10.1063/1.2737444) |

`Wilson1955` is a book (McGraw-Hill; Dover reprint 1980).
`QChemPHVA` is the Q-Chem 6.3 manual, § 10.7.4, read online.

## What rests on full text, and what rests on an abstract

| key | record checked against | content |
|---|---|---|
| `Eckart1935` | Crossref **and** the APS record | abstract read; the citation was **narrowed** — it is cited for the separation of vibration from rotation, and no longer for the stationary-point condition, which is Wilson's |
| `Head1997` | Crossref | added 2026-09-21 to **correct an attribution**: the contract had credited Li & Jensen with a method Head originated |
| `LiJensen2002` | Crossref | names and analyses the method rather than originating it |
| `Ghysels2007` | Crossref | **abstract only** — see the gap below |
| `Ghysels2008` | read directly | closes most of that gap: the same group, stating the three-versus-six fact outright |
| `Besley2008` | Crossref | full text read |
| `Tao2021` | Crossref | full text read |
| `Vester2024` | Europe PMC | full text read |
| `QChemPHVA` | — | page read in a browser; the HTML is JavaScript-built, so a plain fetch returns navigation only |

**The one remaining gap.** `Ghysels2007` (the MBH paper proper) is still known
here only through its abstract. It is cited for two things — that MBH extends
Head's approach, and that its stated aim is to avoid artificial imaginary
frequencies while keeping track of global translation and rotation — and both
are in the abstract. `Ghysels2008` corroborates the underlying fact from the
same group. **No claim in the contract depends on a number or a derivation from
inside the 2007 paper.** If a future edit needs one, read the article first: an
abstract is not a source for a derivation.
