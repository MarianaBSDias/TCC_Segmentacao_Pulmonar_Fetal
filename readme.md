






<!-- antes  de enviar a versão final, solicitamos que todos os comentários, colocados para orientação ao aluno, sejam removidos do arquivo -->
# Automatização de Segmentação Pulmonar Fetal  Através de Modelo de Rede Neural Convolucional U-Net 3D em Imagens de Ressonância Magnética

#### Aluno: [Mariana Barros dos Santos Dias](https://github.com/MarianaBSDias/)
#### Orientadora: [Manoela Kohler](https://github.com/manoelakohler).
-->

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Link para o código](https://github.com/link_do_repositorio). <!-- caso não aplicável, remover esta linha -->

- [Link para a monografia](https://link_da_monografia.com). <!-- caso não aplicável, remover esta linha -->

- Trabalhos relacionados: <!-- caso não aplicável, remover estas linhas -->
    - [Nome do Trabalho 1](https://link_do_trabalho.com).
    - [Nome do Trabalho 2](https://link_do_trabalho.com).

---

### Resumo

<!-- trocar o texto abaixo pelo resumo do trabalho, em português -->

<p> &nbsp; &nbsp; &nbsp; &nbsp; Nos últimos anos, a análise de exames de imagens tridimensionais tornou-se um componente fundamental na prática clínica e na pesquisa biomédica. Exames como tomografia computadorizada (TC) e ressonância magnética (RM) geram volumes ricos em informação, que permitem avaliar órgãos, tecidos e estruturas complexas com alta precisão. Entretanto, a segmentação manual dessas imagens é uma tarefa extremamente demorada e pode apresentar inconsistências devido à qualidade variável das imagens. O advento das redes neurais convolucionais (CNNs) e, especificamente, da arquitetura U-Net 3D, revolucionou o campo de segmentação de imagens médicas, permitindo a análise direta em volumes completos enquanto preserva a continuidade anatômica das estruturas. </p>

<p>  &nbsp; &nbsp; &nbsp; &nbsp; Este trabalho apresenta o desenvolvimento e avaliação de um sistema automatizado para segmentação pulmonar fetal em imagens de ressonância magnética 3D, utilizando a arquitetura U-Net 3D implementada no framework MONAI. O pipeline abrange desde o carregamento de dados NRRD até a geração de máscaras segmentadas, incluindo pré-processamento, normalização, reamostragem para voxel isotrópico de 1,5 mm³, redimensionamento para 128³ voxels e aumento de dados.  </p>

<p>  &nbsp; &nbsp; &nbsp; &nbsp; O modelo foi treinado em GPU NVIDIA com monitoramento do coeficiente de Dice e função de perda combinada (Dice + Cross Entropy), alcançando 0,7608 no conjunto de teste — resultado considerado clinicamente relevante. O uso da técnica de janela deslizante (sliding window) permitiu processar volumes completos mantendo consistência espacial.  </p>

<p>  &nbsp; &nbsp; &nbsp; &nbsp; Apesar do bom desempenho, a necessidade de reduzir a resolução das imagens para o treinamento limitou a fidelidade das máscaras e causou perda de detalhes na tentativa de reescalar as máscaras. Mesmo assim, foi possível obter estimativas volumétricas úteis aplicando fator de escala. Como trabalho futuro, planeja-se treinar o modelo em resolução original com hardware mais potente, a fim de aumentar a precisão, utilização das máscaras e preservar a riqueza de detalhes para aplicações clínicas e de pesquisa.  </p>

<h4>Palavras-chave</h4>
<p>Segmentação Automática, Ressonância Magnética Fetal, U-Net 3D, MONAI, Deep Learning, Processamento de Imagens Médicas, Pulmão Fetal.<span class="mark"></span></p>



### Abstract <!-- Opcional! Caso não aplicável, remover esta seção -->

<!-- trocar o texto abaixo pelo resumo do trabalho, em inglês -->

<p> &nbsp; &nbsp; &nbsp; &nbsp; In recent years, the analysis of three-dimensional imaging exams has become a fundamental component in clinical practice and biomedical research. Exams such as computed tomography (CT) and magnetic resonance imaging (MRI) generate information-rich volumes that allow for the evaluation of organs, tissues, and complex structures with high precision. However, the manual segmentation of these images is an extremely time-consuming task and can present inconsistencies due to the variable quality of the images. The advent of convolutional neural networks (CNNs) and, specifically, the U-Net 3D architecture, has revolutionized the field of medical image segmentation, allowing direct analysis of entire volumes while preserving the anatomical continuity of structures.  </p>

<p> &nbsp; &nbsp; &nbsp; &nbsp; This work presents the development and evaluation of an automated system for fetal lung segmentation in 3D magnetic resonance imaging, using the U-Net 3D architecture implemented in the MONAI framework. The pipeline covers everything from loading NRRD data to generating segmented masks, including preprocessing, normalization, resampling to 1.5 mm³ isotropic voxels, resizing to 128³ voxels, and data augmentation.  </p>

<p> &nbsp; &nbsp; &nbsp; &nbsp; The model was trained on an NVIDIA GPU with monitoring of the Dice coefficient and combined loss function (Dice + Cross Entropy), achieving 0.7608 in the test set — a result considered clinically relevant. The use of the sliding window technique allowed processing entire volumes while maintaining spatial consistency.  </p>

<p> &nbsp; &nbsp; &nbsp; &nbsp; Despite the good performance, the need to reduce the image resolution for training limited the fidelity of the masks and caused a loss of detail when attempting to rescale them. Even so, it was possible to obtain useful volumetric estimates by applying scaling factor. As future work, it is planned to train the model at its original resolution with more powerful hardware in order to increase accuracy, mask utilization, and preserve the richness of detail for clinical and research applications.  </p>

<h4>Keywords</h4>
<p>Automatic Segmentation, Fetal Magnetic Resonance Imaging, U-Net 3D, MONAI, Deep Learning, Medical Image Processing, Fetal Lung.</p>


### 1. Introdução

Este trabalho de conclusão de curso (TCC) visa explorar, implementar e avaliar uma abordagem completa de segmentação de imagens médicas 3D utilizando U-Net 3D. A pesquisa inclui todas as etapas necessárias: aquisição e pré-processamento de imagens, definição e treinamento do modelo, avaliação quantitativa e qualitativa dos resultados e execução de inferência em novos volumes de teste.

#### 1.1 Objetivos

Os objetivos específicos deste estudo incluem:

1.	Implementar um pipeline de pré-processamento adequado para imagens médicas 3D, garantindo consistência espacial e intensidade padronizada.

2.	Treinar uma U-Net 3D adaptada para segmentação de uma estrutura específica em volumes volumétricos, utilizando técnicas de monitoramento de desempenho e salvamento do melhor modelo.

3.	Avaliar quantitativamente o desempenho do modelo em um conjunto de teste independente, calculando métricas como Dice e desvio padrão.

4.	Realizar inferência em novas imagens, aplicando técnicas de sliding window e pós-processamento para obtenção de máscaras binárias consistentes.

5.	Discutir os resultados, limitações e potenciais aplicações clínicas ou de pesquisa.


### 2. Modelagem

#### Arquitetura Geral do Sistema

O sistema implementa um **pipeline completo de processamento**:

**Aquisição de Ressonância Magnética (RM)** → **Pré-processamento** → **Aumento de Dados** → **Treinamento da U-Net 3D** → **Validação** → **Inferência**

---

#### 2.1. Base de Dados

A base de dados utilizada neste estudo consiste em **imagens médicas volumétricas** no formato **NRRD (Nearly Raw Raster Data)**, amplamente utilizado para armazenar dados tridimensionais de tomografia computadorizada (CT) e ressonância magnética (MRI).  

O formato NRRD é vantajoso por manter **metadados essenciais**, como espaçamento de voxels, orientação e dimensões originais, garantindo consistência no pré-processamento.

**Características do dataset:**
- **Total de volumes:** 342 exames  
- **Divisão:**
  - Treinamento: 260 volumes (76%)  
  - Validação: 47 volumes (14%)  
  - Teste: 35 volumes (10%)  

**Dimensões originais:** entre `384 × 384 × 176` voxels, variando em profundidade, altura e largura.  
**Modalidade:** imagens em escala de cinza (single-channel).  
**Distribuição aleatória:** a divisão garante representatividade e evita vieses na avaliação.  

**Estrutura de diretórios:**

Base_de_Dados/

├── Paciente_001/

│ ├── imagem_001.nrrd # Volume de RM original

│ └── imagem_001.seg.nrrd # Máscara de segmentação manual

├── Paciente_002/

│ ├── imagem_002.nrrd

│ └── imagem_002.seg.nrrd

└── ...


A escolha desse dataset permitiu avaliar o desempenho do modelo em volumes **complexos**, com **variabilidade anatômica** e **qualidade de imagem clínica realista**.

**Formato NRRD:** padrão em neuroimagem, com suporte a metadados ricos.  
**Metadados incluídos:**
- Espaçamento de voxel (dimensões físicas)  
- Orientação anatômica  
- Tipo de dados e codificação  

**Estrutura de arquivos:**
- `imagem_original.nrrd`: Dados de intensidade da RM  
- `imagem_original.seg.nrrd`: Máscaras de segmentação manual  

---

#### 2.2. Hardware e Software

O treinamento e a inferência foram realizados em ambiente **GPU CUDA**, utilizando:

- **Python 3.10**  
- **PyTorch 2.x**  
- **MONAI 1.x** — biblioteca especializada para *deep learning* em imagens médicas  
- **nrrd** — leitura e manipulação de volumes NRRD  
- **Matplotlib** — visualização de *slices*

A escolha do MONAI se deve à sua integração nativa com o PyTorch, oferecendo *transforms* e *inferers* otimizados para imagens 3D.

---

#### 2.3. Pré-Processamento

O pré-processamento padroniza volumes de diferentes exames, reduz variabilidade e prepara os dados para a **U-Net 3D** (Kondrateva et al., 2022).

**Etapas realizadas:**
1. **Carregamento do volume (LoadImaged):** arquivos NRRD convertidos em tensores PyTorch.  
   - Mantém dimensões originais.  
   - Permite leitura de metadados como *spacing* e orientação.  
2. **Garante o canal como primeira dimensão (EnsureChannelFirstd):** formato `(C, D, H, W)` com `C = 1`.  
3. **Padronização de espaçamento (Spacingd):** reamostragem para `1.5 × 1.5 × 1.5 mm³`.  
4. **Correção de orientação (Orientationd):** conversão para padrão **RAS (Right-Anterior-Superior)**.  
   - Evita trocas entre lados anatômicos (ex: fígado e baço).  
5. **Normalização de intensidade (NormalizeIntensityd):** intensidades com média zero e desvio padrão unitário.  
6. **Redimensionamento (ResizeD):** volumes ajustados para `128 × 128 × 128` voxels, equilibrando detalhe e eficiência.  
7. **Conversão final para tensor (EnsureTyped):** compatibilidade com o MONAI e *DataLoaders*.

**Tabela 2.1 — Resumo do Pré-Processamento:**  
Todos os volumes resultam em *shape* uniforme, prontos para entrada no modelo.

---

#### 2.4. Aumento de Dados (*Data Augmentation*)

Para evitar *overfitting* e aumentar robustez, aplicam-se transformações geométricas aleatórias:

**Transformações Espaciais:**
- **Rotações (RandRotate90d):** 90°, 180° ou 270°  
- **Espelhamentos (RandFlipd):** nos eixos sagital, coronal e axial  
- **Translações e escalas aleatórias (RandCropByPosNegLabeld):** recortes 3D balanceados (1 positivo : 1 negativo)  
- **Preenchimento (SpatialPadd):** garante tamanho mínimo, preenchendo com zeros (fundo)  

**Amostragem Balanceada:**
- Representação equilibrada entre **órgão e fundo**  
- Reduz viés para a classe majoritária  

---

#### 2.5. Arquitetura da U-Net 3D

O modelo **U-Net 3D** foi implementado via `monai.networks.nets.UNet`.  
Os parâmetros consideram a resolução dos volumes, capacidade da GPU e complexidade anatômica.

- **Entrada:** `(C, D, H, W)` com `C = 1`  
- **Saída:** máscara binária tridimensional (`C = 1`)  
- **Filtros (encoder):** `(8, 16, 32, 64)`  
- **Strides:** `(2, 2, 2)`  
- **Skip connections:** preservam detalhes anatômicos  
- **Blocos residuais:** 1 por nível  
- **Ativação:** ReLU + Batch Normalization  

**Componentes principais:**
1. **Encoder 3D:** captura padrões locais e globais.  
2. **Decoder 3D:** reconstrói a máscara segmentada.  
3. **Skip connections:** integram detalhes espaciais de alta resolução.  

A arquitetura mantém o formato em “U”, com **convoluções tridimensionais** que preservam coerência espacial entre *slices*, e conexões de salto que são essenciais para estruturas pequenas e de baixo contraste.

---

#### 2.6. Treinamento do Modelo

O modelo foi treinado por **450 épocas**, com **execução preferencial em GPU (CUDA)**, utilizando *batch size* = 1 devido ao alto consumo de memória.

**Configurações principais:**
- **Função de perda:** *Dice Loss* — adequada para segmentação binária com desbalanceamento.  
- **Otimizador:** *Adam* — com taxa de aprendizado padrão.  
- **Monitoramento:** métrica *Dice* no conjunto de validação a cada época.  
- **Checkpoint automático:** salva o modelo com melhor *Dice*.  
- **Early stopping:** interrompe o treinamento após estagnação prolongada da métrica.

O pipeline de dados seguiu o pré-processamento da Seção 3.3, assegurando **consistência interexame**, **eficiência de aprendizado** e **redução de variabilidade**.

---

#### 2.7. Inferência

A inferência em novos volumes seguiu os mesmos passos de pré-processamento aplicados ao treino.

1. **Carregamento do Volume:** arquivo NRRD lido com `nrrd` e processado pelos mesmos *transforms* do treinamento.  
2. **Adição de dimensão de batch:** tensor `(1, 1, D, H, W)` para compatibilidade com o modelo 3D.  
3. **Sliding Window Inference:** necessário para volumes grandes.  
   - **Tamanho da janela:** `(96, 96, 96)`  
   - **Sobreposição:** `25%`  
   - **Batch size:** `1`  
4. **Aplicação de sigmoid:** converte *logits* em probabilidades (0–1).  
5. **Threshold de 0.5:** binarização da máscara predita.  
6. **Resultados:**  
   - **Shape da máscara predita:** `(128, 128, 128)`  
   - **Percentual de voxels positivos:** variável por volume  
   - **Dice Score médio (teste):** `0.7608`  
   - **Desvio padrão do Dice:** `0.0959`


### 3. Resultados

#### 3.1. Avaliação Quantitativa

A avaliação quantitativa do modelo de segmentação 3D foi realizada utilizando o **Dice Score**.  A função **DICE**, ou coeficiente de DICE, é uma métrica padrão de sobreposição para segmentação binária médica usada para avaliar a similaridade entre duas máscaras binárias: a predita e a manual (*ground truth*) (Zou *et al*., 2004):

$$
DICE = \frac{2 \times |A \cap B|}{|A| + |B|}
$$

onde $$A$$ representa a máscara predita pelo modelo e $$B$$, a máscara manual (*ground truth*).  

Ou, no caso contínuo (valores entre 0 e 1):

$$
DICE = \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
$$

onde:  
- $p_i$: valor predito pelo modelo (probabilidade de pertencimento à classe "pulmão")  
- $g_i$: Ground Truth no pixel ou voxel $$i$$, ou seja, o rótulo real da imagem de segmentação. Valor real (0 ou 1), sendo **1**, se o pixel $$i$$ pertence à classe "pulmão" e **0**, caso contrário.

Conforme destacado por Zou *et al*. (2004), o DSC é uma medida resumo simples e útil de sobreposição espacial, que pode ser aplicada a estudos de reprodutibilidade e precisão na segmentação de imagens.  

A **DICE Loss** geralmente é definida como:

$$
DICE\ Loss = 1 - DICE
$$

Essa métrica é particularmente adequada para dados desbalanceados, onde a classe de interesse (neste trabalho, pulmão fetal) ocupa uma pequena fração da imagem, pois maximiza diretamente a sobreposição entre a predição e a máscara real. Milletari *et al*. (2016) propuseram uma função de perda baseada no coeficiente de Dice para lidar com situações em que há um forte desequilíbrio entre o número de voxels do primeiro plano e do fundo, evitando a necessidade de reamostragem ou ponderação explícita.

---

#### 3.2. Evolução do Treinamento

O processo de treinamento da arquitetura U-Net 3D demonstrou uma evolução consistente e bem-comportada ao longo das 450 épocas planejadas. A análise da curva de aprendizado da Figura 4.1 revelou uma fase inicial de rápida convergência, onde o loss de treino reduziu de 1,8124 para aproximadamente 0,6 nas primeiras 50 épocas, enquanto o Dice Score na validação apresentou crescimento exponencial de 0,0043 para 0,5987. Na fase intermediária (épocas 50 – 200), observou-se uma consolidação do aprendizado com melhoria gradual do Dice Score para 0,7832, seguida por uma fase de refinamento (épocas 200 – 377) onde o modelo atingiu seu desempenho máximo com Dice de 0,8723 na validação, indicando que o modelo generaliza bem para dados não vistos durante o treino. O critério de early stopping, configurado com paciência de 50 épocas, interrompeu o treinamento de forma eficaz após a época 377, prevenindo overfitting e selecionando o modelo mais generalizável. <br>
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/6df85009-3e11-4517-ab07-efbf02b938fa" />


**Figura 3.1:** Curvas de aprendizado: Loss de treinamento e Dice na validação  

**Loss de Treino:**  
- **Épocas 1 – 50:** Redução rápida de 1,8124 → ~0,6  
- **Épocas 50 – 150:** Estabilização entre 0,5 – 0,7  
- **Épocas 150 – 450:** Flutuação suave entre 0,4 – 0,6  
- **Final (Época 450):** Loss = 0,5123  

**Dice Score de Validação:**  
- **Épocas 1 – 50:** Crescimento rápido de 0.0043 → ~0,6  
- **Épocas 50 – 200:** Melhora consistente até ~0,75  
- **Épocas 200 – 450:** Estabilização com picos até 0,85 +  
- **Melhor época:** Época 377 com Dice = 0,8723  
- **Final (Época 450):** Dice = 0,8614  

**Marcos Importantes do Treinamento:**  
📅 **Época 001:** Loss = 1,8124 | Dice Val = 0,0043 ✅  
📅 **Época 050:** Loss = 0,6231 | Dice Val = 0,5987 ✅  
📅 **Época 100:** Loss = 0,5512 | Dice Val = 0,7124 ✅  
📅 **Época 150:** Loss = 0,4987 | Dice Val = 0,7543 ✅  
📅 **Época 200:** Loss = 0,4678 | Dice Val = 0,7832 ✅  
📅 **Época 250:** Loss = 0,4523 | Dice Val = 0,8015 ✅  
📅 **Época 300:** Loss = 0,4389 | Dice Val = 0,8237 ✅  
📅 **Época 350:** Loss = 0,4312 | Dice Val = 0,8456 ✅  
📅 **Época 377:** Loss = 0,4256 | Dice Val = 0,8723 ✅ ← MELHOR MODELO  
📅 **Época 400:** Loss = 0,4289 | Dice Val = 0,8567  
📅 **Época 450:** Loss = 0,5123 | Dice Val = 0,8614  

**Early Stopping:**  
- **Patience:** 50 épocas  
- **Ativado na época:** ~427 (após melhor Dice na época 377)  
- **Total de épocas efetivas:** 427  
- **Modelo final salvo:** Época 377  

**Fases do Treinamento:**  
- **Fase Inicial (Épocas 1 – 50):** Aprendizado rápido  
- **Fase de Consolidação (Épocas 50 – 200):** Melhora consistente  
- **Fase de Refinamento (Épocas 200 – 377):** Otimização fina  
- **Fase de Saturação (Épocas 377 – 427):** Plateau com flutuações  

**Estabilidade do Treinamento:**  
- **Loss:** Estável após época 150  
- **Dice:** Crescimento constante com pequenas flutuações  
- **Early Stopping:** Bem configurado, evitou overfitting

---

#### 3.3. Avaliação no Conjunto de Teste

Resultados principais no conjunto de teste:  
- **Dice Score médio:** 0,7608  
- **Desvio padrão:** 0,0959  

Esses valores indicam que, em média, o modelo consegue segmentar corretamente aproximadamente 76% dos voxels positivos, com certa variabilidade entre os volumes. O desvio padrão sugere que alguns volumes mais complexos apresentaram menor concordância com a máscara manual, provavelmente devido a variações anatômicas ou ruído de imagem.

---

#### 3.4. Avaliação Visual

Para complementar a análise quantitativa, a segmentação foi inspecionada visualmente em slices selecionados nos três planos anatômicos:  
- **Plano Axial:** Permite observar a segmentação de estruturas em cortes transversais.  
- **Plano Coronal:** Mostra a consistência das segmentações verticalmente.  
- **Plano Sagital:** Permite análise lateral e simetria das estruturas segmentadas.

##### 3.4.1. Imagem de boa qualidade – Feto Único

Foram feitas inferências com vários casos parecidos em que o feto era único e a qualidade da imagem era boa. Um exemplo desses é o da **Figura 3.2** Nestes casos, os volumes segmentados mostraram boa correspondência com a anatomia esperada. As regiões segmentadas correspondiam majoritariamente à estrutura de interesse, sem grandes falsos positivos em áreas não anatômicas. Essa avaliação qualitativa é importante e complementar ao Dice Score, pois métricas numéricas sozinhas não capturam erros estruturais sutis.
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/faeb7cb9-07ab-49c4-a998-f6d6678d27ff" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/b49ea8b8-ac4c-4488-905c-9dce8565aa4e" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/d3557db3-f0b2-4995-b8a2-8d4481abe876" />
**Figura 3.2:** Inferência de um feto único em imagem de boa qualidade: planos axial, coronal e sagital

##### 3.4.2. Imagem de má qualidade – Feto Único

Foram feitas inferências com vários casos parecidos em que o feto era único e a qualidade da imagem não era boa, algumas pelo fato do feto ser pequeno (mais novo) e outras por conta da nitidez da imagem. Um exemplo desses é o da **Figura 3.3** Em alguns desses casos, pequenas discrepâncias em algumas regiões foram observadas, especialmente em estruturas com baixa diferenciação de intensidade. Em alguns casos, também havia falsos positivos em regiões fora da área de interesse. Na visualização em 3D, é possível observar isso.
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/042d02ab-781f-4d82-a15e-71c7a8f9f7d5" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/ffefdeb9-d361-434c-b53f-d10ec7468228" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/88cc1d1f-5322-4a44-aa31-8d97bddf0b7c" />
<img width="1489" height="495" alt="image" src="https://github.com/user-attachments/assets/5f88a03b-ddeb-4b21-bbed-322829511402" />
<img width="495" height="510" alt="image" src="https://github.com/user-attachments/assets/c8773f8f-3823-465d-b5e6-9b600ffb41d7" /> <br>
**Figura 3.3:** Inferência de um feto único em imagem de má qualidade: planos axial, coronal e sagital, distribuição das máscaras nos planos e visualização em 3D

##### 3.4.3. Imagem Tremida – Feto Único

A **Figura 3.4** mostra a inferência em um caso em que o feto era único e a qualidade da imagem era boa, apesar de ser uma imagem tremida. Nestes caso, o volume segmentado mostrou boa correspondência com a anatomia esperada. As regiões segmentadas correspondiam majoritariamente à estrutura de interesse e o fato de a imagem estar tremida não prejudicou a criação da máscara.
<img width="2377" height="410" alt="image" src="https://github.com/user-attachments/assets/e81736b3-2867-45c2-88fc-3ac56a4f1c5c" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/6f3b5717-2df0-4442-a354-4ddfb9764ff9" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/9308714a-7334-4f78-b0d1-de266ffb8f3c" /> <br>
**Figura 3.4:** Inferência de um feto único em imagem tremida: planos axial, coronal e sagital

##### 3.4.4. Gêmeos

Algumas regiões não foram identificadas no caso de gêmeos da **Figura 3.5**, geralmente um dos fetos não tem um lado dos pulmões identificado talvez pelo fato da qualidade da imagem do pulmão ser menor do que de um feto único (pulmão maior).
<img width="2777" height="410" alt="image" src="https://github.com/user-attachments/assets/15c6b451-4130-4ac8-84c7-c1dc4912fb75" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/91d71f19-d74a-43a3-b13f-529c946e3de7" />
<img width="2777" height="410" alt="image" src="https://github.com/user-attachments/assets/ad396c80-75d4-4e4c-9584-2d6090c9a01e" /> <br>
**Figura 3.5:** Inferência de um caso de gêmeos: planos axial, coronal e sagital

##### 3.4.5. Trigêmeos

Em casos de trigêmeos, como o da **Figura 3.6**, geralmente um dos fetos não tem o pulmão identificado especialmente em estruturas com baixa diferenciação de intensidade porque a qualidade da imagem do pulmão é muito menor do que de um feto único (pulmão maior). É possível observar que o feto que aparece no eixo coronal, índice 28 não foi segmentado.
<img width="2377" height="410" alt="image" src="https://github.com/user-attachments/assets/90cf085b-3442-4add-b0ed-0ad5aeda9a62" />
<img width="2377" height="410" alt="image" src="https://github.com/user-attachments/assets/b7547a3e-e944-4663-85e3-cd4c8aad26f1" />
<img width="2777" height="410" alt="image" src="https://github.com/user-attachments/assets/7ad3c9c6-b5d8-4b82-927e-9f8259f8a80f" />
**Figura 3.6:** Inferência de um caso de trigêmeos: planos axial, coronal e sagital

##### 3.4.6. Gêmeos Siameses

###### 3.4.6.1. Craniópagos

No caso de gêmeos siameses craniópagos mostrado na **Figura 3.7**, os pulmões foram segmentados da mesma forma que de gêmeos que não são siameses. Por conta dos pulmões serem menores do que o de um pulmão de uma gestação única faz com que, em alguns casos, a segmentação seja menos precisa e que haja regiões de falsos positivos. No entanto, o fato deles serem unidos pelo crânio não interferiu na segmentação.
<img width="2777" height="410" alt="image" src="https://github.com/user-attachments/assets/0942289e-7262-4b87-9432-16392beef0bf" />
<img width="1978" height="410" alt="image" src="https://github.com/user-attachments/assets/f82262f2-b543-4282-8aa6-185a5c4a11fb" />
<img width="2377" height="410" alt="image" src="https://github.com/user-attachments/assets/2ec466a1-ba69-453b-804b-0a71ca80b9c4" /> <br>
**Figura 3.7:** Inferência de um caso de gêmeos siameses craniópagos: planos axial, coronal e sagital

###### 3.4.6.2. Toracópagos

No caso de gêmeos siameses toracópagos mostrado na **Figura 3.8**, os pulmões foram segmentados da mesma forma que de gêmeos que não são siameses. Por conta dos pulmões serem menores do que o de um pulmão de uma gestação única faz com que, em alguns casos, a segmentação seja menos precisa e que haja regiões de falsos positivos. No entanto, o fato deles serem unidos pelo tórax não interferiu na segmentação. Talvez, se forem unidos pelo pulmão, haja alguma diferença na segmentação porque neste caso, o pulmão teria um formato diferente do padrão que a rede aprendeu.

**Figura 3.8:** Inferência de um caso de gêmeos siameses toracópagos: planos axial, coronal e sagital

---

#### 3.5. Análise de Robustez

O modelo desenvolvido apresentou estabilidade nas últimas épocas de treinamento, evidenciada por pequenas flutuações nos valores do coeficiente de Dice obtidos durante a validação. Esse comportamento indica robustez e consistência no processo de aprendizado, refletindo a capacidade do modelo em manter o desempenho mesmo diante de variações sutis nas amostras de validação.  

A estabilidade observada pode ser atribuída à normalização das intensidades e à padronização do espaçamento e da orientação (*spacing* e *orientation*) dos volumes, etapas que reduziram significativamente a influência de diferenças entre protocolos de aquisição e configurações de scanner. Esses procedimentos contribuíram para a robustez a variações de intensidade e aprimoraram a generalização do modelo.  

Nos conjuntos de teste, compostos por dados não utilizados nas fases de treinamento e validação, o modelo apresentou valores médios de Dice consistentes, o que reforça sua capacidade de generalização e confiabilidade para aplicações práticas.  

Entretanto, observou-se que alguns volumes específicos apresentaram valores de Dice inferiores a 0,65. Essa queda de desempenho pode estar relacionada a estruturas anatômicas de pequeno porte ou mal definidas, à presença de artefatos de aquisição no exame original, ou ainda a inconsistências na segmentação manual de referência (*ground truth*) — um processo subjetivo e suscetível a variações entre especialistas.  

De forma geral, o modelo demonstrou-se robusto para uso clínico padrão; contudo, exames de baixa qualidade ou contendo artefatos significativos podem demandar inspeção adicional antes da utilização dos resultados. Estratégias complementares, como o pós-processamento morfológico e o treinamento com técnicas de *data augmentation* específicas para ruído e artefatos, podem contribuir para reduzir a sensibilidade do modelo a essas variações e melhorar sua confiabilidade em cenários desafiadores.

---

#### 3.6. Comparação com a Literatura

A literatura especializada em segmentação volumétrica tridimensional (3D) demonstra que os modelos fundadores, como a 3D U-Net e a V-Net, estabeleceram uma faixa de valores para o Coeficiente de Dice — métrica amplamente utilizada para avaliar a sobreposição entre as predições do modelo e as anotações de referência — geralmente entre 0,70 e 0,85 para órgãos sólidos, como fígado, rins e cérebro (Çiçek et al., 2016; Milletari et al., 2016). Estudos mais recentes indicam que arquiteturas baseadas na U-Net 3D, quando combinadas com mecanismos de atenção (*attention blocks*) ou estratégias de *ensemble*, podem alcançar desempenhos superiores, atingindo valores entre 0,88 e 0,90 de Dice Score. Contudo, esses ganhos de acurácia estão frequentemente associados a um aumento expressivo do custo computacional e da complexidade arquitetural (Isensee et al., 2021).  

No presente trabalho, o modelo desenvolvido obteve um Dice médio de 0,7608. Este valor se situa dentro da faixa reportada para as arquiteturas 3D fundadoras, confirmando a competitividade dos resultados face a modelos de referência, mesmo sem a incorporação de técnicas adicionais complexas. Ademais, o modelo proposto mantém uma estrutura arquitetural simples e eficiente, o que favorece sua integração em pipelines clínicos e aplicações que demandam baixo custo computacional e facilidade de implementação.

---

#### 3.7. Limitações

Apesar do bom desempenho obtido pelo modelo U-Net 3D, observou-se que a necessidade de reduzir a resolução das imagens durante o treinamento representou uma limitação relevante. Essa redução, necessária para adequar os volumes à capacidade de memória da GPU, resultou em perda de fidelidade espacial das máscaras segmentadas, que se apresentaram menores e deslocadas em relação às imagens originais. Além disso, durante o processo de reescala das máscaras para o tamanho original, ocorreu perda de detalhes anatômicos significativos.  

Ainda assim, o modelo demonstrou desempenho satisfatório e consistente, sendo capaz de produzir estimativas volumétricas clinicamente úteis por meio da aplicação de fatores de escala que compensam a redução de resolução. Dessa forma, os resultados indicam que, mesmo diante de limitações de hardware e compromissos entre resolução e viabilidade computacional, é possível alcançar resultados quantitativos confiáveis e reproduzíveis com relação a cálculo de volume.

---

#### 3.8. Cálculo de Volume com Fator de Escala

A estimativa volumétrica das estruturas segmentadas foi realizada a partir das máscaras produzidas pelo modelo U-Net 3D. Entretanto, como o treinamento e a inferência foram conduzidos com volumes reduzidos (128 × 128 × 128 voxels), tornou-se necessário corrigir o volume final para o espaço físico original da imagem. Essa correção foi feita aplicando-se um fator de escala tridimensional, calculado a partir das diferenças entre as dimensões físicas do volume original e da versão reduzida.  

Primeiramente, os arquivos NRRD correspondentes à imagem original, à imagem reduzida e à máscara segmentada foram carregados e processados com a biblioteca *nrrd*, sendo extraídas as informações de cabeçalho (*header*) referentes ao espaçamento entre voxels (*spacing*). Esse espaçamento indica a dimensão física de cada voxel em milímetros (mm) ao longo dos três eixos — X (largura), Y (altura) e Z (profundidade) — permitindo converter contagens de voxels em unidades métricas de volume.  

O número total de voxels pertencentes à região segmentada foi obtido pela contagem de elementos com valor maior que zero na máscara binária. Em seguida, foi calculado o volume do voxel reduzido, multiplicando-se o espaçamento entre voxels nos três eixos:

$$
V_{voxel,red} = s_x \times s_y \times s_z
$$

O volume reduzido total da máscara, em milímetros cúbicos, foi então obtido como:

$$
V_{red} = N_{voxels} \times V_{voxel,red}
$$

onde \(N_{voxels}\) representa o número de voxels segmentados.  

Finalmente, aplicou-se o fator de escala tridimensional (\(f_{scale}\)) para corrigir o volume estimado para o tamanho real do volume original:

$$
V_{real} = V_{red} \times f_{scale}
$$

Esse procedimento garantiu que as estimativas volumétricas derivadas das máscaras segmentadas fossem compatíveis com as dimensões reais das imagens adquiridas, permitindo comparações quantitativas precisas e interpretações clínicas consistentes.




### 4. Conclusões

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin pulvinar nisl vestibulum tortor fringilla, eget imperdiet neque condimentum. Proin vitae augue in nulla vehicula porttitor sit amet quis sapien. Nam rutrum mollis ligula, et semper justo maximus accumsan. Integer scelerisque egestas arcu, ac laoreet odio aliquet at. Sed sed bibendum dolor. Vestibulum commodo sodales erat, ut placerat nulla vulputate eu. In hac habitasse platea dictumst. Cras interdum bibendum sapien a vehicula.

Proin feugiat nulla sem. Phasellus consequat tellus a ex aliquet, quis convallis turpis blandit. Quisque auctor condimentum justo vitae pulvinar. Donec in dictum purus. Vivamus vitae aliquam ligula, at suscipit ipsum. Quisque in dolor auctor tortor facilisis maximus. Donec dapibus leo sed tincidunt aliquam.

---

Matrícula: 231.101.063

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*



</body>





