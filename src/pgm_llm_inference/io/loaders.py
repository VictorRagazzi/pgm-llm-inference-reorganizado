import gzip
import tempfile
from pathlib import Path

from pgm_llm_inference.core.conversion import convert_pgmpy_model, parse_dne_to_pgmpy
from pgm_llm_inference.models import BayesianNetwork


def load_network(path: str, context_type=None, llm_fn=None) -> tuple:
    """
    Carrega um arquivo de rede Bayesiana (.bif, .bif.gz, .xdsl, .xdsl.gz,
    .net, .dne, .dne.gz) e retorna (BayesianNetwork, context_str).

    context_type e llm_fn são mantidos na assinatura para compatibilidade,
    mas o sistema de contexto MAP foi removido. context_type deve ser None;
    context retornado é sempre "".
    """
    from pgmpy.readwrite import BIFReader, XDSLReader, NETReader

    path = Path(path)
    suffixes = "".join(path.suffixes)

    def open_temp(p: Path, force_suffix: str | None = None) -> str:
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="ISO-8859-1") as f:
                content = f.read()
        else:
            content = p.read_text(encoding="ISO-8859-1")

        suffix = force_suffix if force_suffix else p.suffix
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=suffix, delete=False, encoding="ISO-8859-1"
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    if suffixes in [".bif", ".bif.gz"]:
        model = BIFReader(open_temp(path)).get_model()
    elif suffixes in [".xdsl", ".xdsl.gz"]:
        model = XDSLReader(open_temp(path)).get_model()
    elif suffixes in [".net", ".net.gz", ".dsc", ".dsc.gz"]:
        model = NETReader(open_temp(path)).get_model()
    elif suffixes in [".dne", ".dne.gz"]:
        model = parse_dne_to_pgmpy(open_temp(path, force_suffix=".dne"))
    else:
        raise ValueError(
            "Supported formats: .bif(.gz), .xdsl(.gz), .net(.gz), .dsc(.gz), .dne(.gz)"
        )

    converted_model = convert_pgmpy_model(model)
    return converted_model, ""
