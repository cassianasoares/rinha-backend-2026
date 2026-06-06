# Rinha Backend 2026 – Fraud Detection API

API para ***detecção de fraude em transações de cartão*** usando **FAISS com IVF + Quantization (QT_8bit)**.  
O projeto segue princípios de Spec-Driven Development e foi desenhado para rodar em infraestrutura extremamente limitada (máx. 1 CPU e 350 MB de memória), mantendo alta performance em busca vetorial.

---

## 🚀 Descrição
- Exposição de endpoints para verificar se a API está pronta (`/ready`) e calcular score de fraude (`/score`).
- Vetores de referência são pré-processados e armazenados em disco (`faiss_index.bin`, `labels.npy`) para inicialização rápida.
- Balanceamento de carga simples via Nginx (round-robin) entre duas instâncias da API.

---

## 🛠️ Stack Tecnológica
- **Linguagem:** Python 3.12
- **Bibliotecas:**  
  - FAISS (Index IVF + Quantization QT_8bit)  
  - NumPy  
- **Framework:** FastAPI + Pydantic
- **Infraestrutura:**  
  - Docker (linux/amd64)  
  - docker-compose (2 instâncias da API, 1 CPU / 350 MB máx.)  
  - Nginx como load balancer (porta 9999)

---

## ⚙️ Padrões de Código
- Nomes descritivos para variáveis e funções.  
- Repositórios para abstrair acesso a dados.  
- Uso de mocks/stubs para dependências externas.  
- Logs estruturados para auditoria.  
- Sem comentários desnecessários no código.  

---

## 🔄 Pré-processamento
- Carregamento de `references.json.gz` → transformação em NumPy array (`float32` ou `float16`).  
- Normalização de campos de transação via `normalization.json`.  
- Conversão para FAISS index com **IVF + Quantization (QT_8bit)**.  
- Salvamento em disco (`faiss_index.bin`, `labels.npy`) para evitar reprocessamento a cada inicialização.  

---

## ⚖️ Trade-offs e Limitações
Para alcançar **3.000.000 vetores** em ambiente restrito (1 CPU e 350 MB RAM), foi necessário:

- **IVF (Inverted File Index):**  
  Permite dividir o espaço vetorial em clusters, reduzindo a busca a apenas uma fração dos vetores.  
  → Trade-off: menor precisão em comparação ao `IndexFlatL2`, mas ganho enorme em velocidade e memória.

- **Quantization (QT_8bit):**  
  Compressão dos vetores para 8 bits, reduzindo drasticamente o uso de memória.  
  → Trade-off: perda de precisão nos cálculos de distância, mas viabiliza rodar milhões de vetores em memória limitada.

- **Float16 em vez de Float32:**  
  Redução de tamanho dos arrays para metade.  
  → Trade-off: menor precisão numérica, mas essencial para caber dentro do limite de 350 MB.

Essas escolhas foram necessárias para equilibrar **performance, memória e precisão**, garantindo que o sistema pudesse escalar para milhões de vetores mesmo em hardware modesto.

---

## 📌 Conclusão
Este projeto demonstra como é possível implementar busca vetorial em larga escala com FAISS, mesmo em ambientes com recursos extremamente limitados.  
O uso de **IVF + QT_8bit** foi a chave para atingir 3 milhões de vetores sem ultrapassar os limites de CPU e memória.