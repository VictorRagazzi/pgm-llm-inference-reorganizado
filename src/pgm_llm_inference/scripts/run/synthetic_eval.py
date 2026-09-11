import time
import random
import pandas as pd
from pathlib import Path

# pgmpy para amostragem
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling

# scikit-learn para métricas
from sklearn.metrics import accuracy_score, f1_score

# Imports do seu projeto
from pgm_llm_inference.experiment.experiment import run_max_product
from pgm_llm_inference.experiment.config import ExperimentConfig
from pgm_llm_inference.mpe.cache import load_or_compile
from pgm_llm_inference.mpe import infer_from_compiled, CompiledSemanticMessages
from pgm_llm_inference.experiment.llm_factory import build_llm_fn, get_model_name
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.paths import dataset_path, metadata_path, relationship_path

def generate_synthetic_ground_truth(bif_path: str, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Gera N amostras a partir da rede bayesiana original."""
    # print(f">>> [SAMPLING] Gerando {n_samples} amostras sintéticas de {bif_path}...")
    reader = BIFReader(bif_path)
    model = reader.get_model()
    
    sampler = BayesianModelSampling(model)
    # forward_sample respeita a topologia e as CPTs exatas da rede
    samples = sampler.forward_sample(size=n_samples, seed=seed, show_progress=False)
    return samples

def evaluate_synthetic_pipeline(
    network,
    compiled: CompiledSemanticMessages,
    bif_path: str,
    evidence_percentages: list[float] = [0.2, 0.4, 0.6],
    n_samples: int = 100, # Reduzido para teste; use 1000 em prod
    seed: int = 42,
) -> pd.DataFrame:
    """Roda a avaliação sintética contra CPT Exato e LLM Semântico."""
    
    # 1. Gerar o Ground Truth
    gt_df = generate_synthetic_ground_truth(bif_path, n_samples, seed)
    variables = list(gt_df.columns)
    
    results = []
    
    for pct in evidence_percentages:
        n_evidence = max(1, int(len(variables) * pct))
        # print(f"\n>>> Avaliando {pct*100}% de evidência ({n_evidence} variáveis observadas)")
        
        y_true_all = []
        y_pred_cpt_all = []
        y_pred_llm_all = []
        
        time_cpt_total = 0.0
        time_llm_total = 0.0
        
        for idx, row in gt_df.iterrows():
            # a. Selecionar aleatoriamente variáveis de evidência
            random.seed(seed + idx)
            evidence_vars = random.sample(variables, n_evidence)
            hidden_vars = [v for v in variables if v not in evidence_vars]
            
            evidence = {v: str(row[v]) for v in evidence_vars}
            ground_truth = {v: str(row[v]) for v in hidden_vars}
            
            # b. Inferência MPE via CPT Exato
            t0 = time.time()
            mpe_result = run_max_product(
                network=network,
                query_vars=hidden_vars,
                evidence=evidence
            )
            time_cpt_total += (time.time() - t0)
            pred_cpt = mpe_result["map_assignment"]
            
            # c. Inferência MPE via LLM (Circuito Semântico)
            t1 = time.time()
            llm_predictions, _, _ = infer_from_compiled(
                compiled=compiled,
                evidence=evidence,
            )
            time_llm_total += (time.time() - t1)
            
            # Alinhamento das predições com o Ground Truth (variável por variável)
            for v in hidden_vars:
                y_true_all.append(ground_truth[v])
                y_pred_cpt_all.append(pred_cpt.get(v, "UNKNOWN"))
                y_pred_llm_all.append(llm_predictions.get(v, "UNKNOWN"))
        
        # d. Calcular Métricas Agregadas
        dataset_name = Path(bif_path).name
        
        # Métricas CPT
        acc_cpt = accuracy_score(y_true_all, y_pred_cpt_all)
        f1_cpt = f1_score(y_true_all, y_pred_cpt_all, average="weighted", zero_division=0)
        
        # Métricas LLM
        acc_llm = accuracy_score(y_true_all, y_pred_llm_all)
        f1_llm = f1_score(y_true_all, y_pred_llm_all, average="weighted", zero_division=0)
        
        results.append({
            "Rede": dataset_name,
            "% Evidência": f"{int(pct*100)}%",
            "Método": "CPT Exato",
            "Acurácia": round(acc_cpt, 4),
            "F1-Weighted": round(f1_cpt, 4),
            "Tempo (s)": round(time_cpt_total, 2)
        })
        
        results.append({
            "Rede": dataset_name,
            "% Evidência": f"{int(pct*100)}%",
            "Método": "LLM Semântico",
            "Acurácia": round(acc_llm, 4),
            "F1-Weighted": round(f1_llm, 4),
            "Tempo (s)": round(time_llm_total, 2)
        })

    # Output no formato de tabela
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RESULTADOS DA AVALIAÇÃO SINTÉTICA (GROUND TRUTH)")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df


def main():
    datasets = [
        "gonorrhoeae.bif",
        "adhd.bif",
        "insurance.bif",
        "crimescene.bif",
        "cryptocurrency.bif",
        # "sachs.bif",
    ]

    cfg = ExperimentConfig(
        dataset_name="",
        use_real_llm=False,
        use_local_llm=True,
    )

    llm_fn = build_llm_fn(use_real_llm=cfg.use_real_llm, use_local_llm=cfg.use_local_llm)
    model_name = get_model_name(cfg.use_real_llm)

    for name in datasets:
        print(f"\n{'=' * 60}")
        print(f">>> Dataset: {name}")
        print(f"{'=' * 60}")
        cfg.dataset_name = name

        # --- Carregamento da rede ---
        path = dataset_path(name)
        network = load_network(path)

        compiled = load_or_compile(
            dataset_name=name,
            model_name=model_name,
            network=network,
            bif_path=path,
            metadata_path=metadata_path(name),
            relationship_path=relationship_path(name),
            llm_fn=llm_fn,
            use_real_llm=cfg.use_real_llm,
        )
        # print(f">>> [COMPILE] ✓ {len(compiled.messages)} mensagens compiladas.")

        # --- INFERÊNCIA: roda para cada evidência — sem LLM no bucket ---
        # Substituí o laço run_batch() pelo nosso novo pipeline de avaliação
        # print(f"\n>>> [INFERENCE] Iniciando pipeline de avaliação sintética...")
        
        evaluate_synthetic_pipeline(
            network=network,
            compiled=compiled,
            bif_path=str(path),
            evidence_percentages=[0.2, 0.4, 0.6], # Parametrizável (ex: 20%, 40%, 60%)
            n_samples=100,                        # 100 para teste inicial. Mude para 1000 em prod!
            seed=42,
        )
        
        # Opcional: Salvar em CSV para manter um log histórico dos experimentos
        # output_csv = path.parent / f"synthetic_eval_{name.split('.')[0]}.csv"
        # df_resultados.to_csv(output_csv, index=False)
        # print(f"\n>>> Resultados salvos em: {output_csv}")

if __name__ == "__main__":
    main()
