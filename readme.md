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


1.1.
Objetivos

Os objetivos específicos deste estudo incluem:

1.	Implementar um pipeline de pré-processamento adequado para imagens médicas 3D, garantindo consistência espacial e intensidade padronizada.

2.	Treinar uma U-Net 3D adaptada para segmentação de uma estrutura específica em volumes volumétricos, utilizando técnicas de monitoramento de desempenho e salvamento do melhor modelo.

3.	Avaliar quantitativamente o desempenho do modelo em um conjunto de teste independente, calculando métricas como Dice e desvio padrão.

4.	Realizar inferência em novas imagens, aplicando técnicas de sliding window e pós-processamento para obtenção de máscaras binárias consistentes.

5.	Discutir os resultados, limitações e potenciais aplicações clínicas ou de pesquisa.


### 2. Modelagem

O sistema implementa um pipeline completo de processamento:

Aquisição de Ressonância Magnética (RM) → Pré-processamento → Aumento de Dados → Treinamento da U-Net 3D → Validação → Inferência


2.1.
Base de Dados

A base de dados utilizada neste estudo consiste em imagens médicas volumétricas no formato NRRD (Nearly Raw Raster Data), um formato amplamente utilizado para armazenar dados tridimensionais de tomografia computadorizada (CT) e ressonância magnética (MRI). O formato NRRD é vantajoso para pesquisas, pois mantém metadados essenciais, como espaçamento de voxels, orientação e dimensões originais, garantindo consistência no pré-processamento.

Características do dataset:                  <br>
•	Total de volumes: 342 exames.            <br>
•	Divisão:                                 <br>
&nbsp;    o	Treinamento: 260 volumes (76%).      <br>
&nbsp;    o	Validação: 47 volumes (14%).         <br>
&nbsp;    o	Teste: 35 volumes (10%).             <br>

•	Dimensões originais: entre 384 × 384 × 176 voxels, variando em profundidade (slices), altura e largura.
•	Modalidade: imagens em escala de cinza (single-channel), representando densidade ou intensidade do tecido.
•	Distribuição aleatória: a divisão foi realizada garantindo representatividade para cada conjunto, evitando vieses na avaliação do modelo.
•	Estrutura de diretórios suportada:
Base_de_Dados/ <br>
├── Paciente_001/
│   ├── imagem_001.nrrd            # Volume de RM original
│   └── imagem_001.seg.nrrd      # Máscara de segmentação manual
├── Paciente_002/
│   ├── imagem_002.nrrd
│   └── imagem_002.seg.nrrd
└── ...
A escolha desse dataset permitiu avaliar o desempenho do modelo em volumes complexos, com variabilidade anatômica e de qualidade de imagem típica de exames clínicos.
Formato NRRD: Formato padrão para neuroimagem com suporte a metadados ricos
Metadados incluídos:
•	Espaçamento de voxel (dimensões físicas)
•	Orientação anatômica
•	Tipo de dados e codificação

Estrutura de arquivos:
•	imagem_original.nrrd: Dados de intensidade da RM
•	imagem_original.seg.nrrd: Máscaras de segmentação manual



2.2.
Hardware e Software
O treinamento e a inferência foram realizados em um ambiente de GPU CUDA, utilizando:
•	Python 3.10
•	PyTorch 2.x
•	MONAI 1.x (biblioteca especializada para deep learning em imagens médicas)
•	nrrd para leitura de volumes NRRD
•	Matplotlib para visualização de slices
A escolha do MONAI se deve à sua integração nativa com PyTorch, fornecendo transforms e inferers prontos para imagens 3D.

2.3.
Pré-Processamento

O pré-processamento é uma etapa crítica para padronizar volumes de diferentes exames, reduzir variabilidade e preparar os dados para a U-Net 3D (Kondrateva et al., 2022). As etapas realizadas foram:

1.	Carregamento do volume (LoadImaged): arquivos NRRD foram convertidos em tensores PyTorch. Este passo assegura que os dados possam ser manipulados de forma eficiente em pipelines de aprendizado profundo.
Considerações técnicas:
•	Mantém dimensões originais do volume para evitar distorção espacial.
•	Permite leitura de metadados como spacing e orientação.

2.	Garante o canal como primeira dimensão (EnsureChannelFirstd): o modelo espera entrada no formato (C, D, H, W), sendo C = 1 (volumes são monocanais). Essa transformação evita erros de dimensionalidade e garante compatibilidade com camadas convolucionais 3D.

3.	Padronização de espaçamento (Spacingd): todos os volumes foram reamostrados para 1,5 × 1,5 × 1,5 mm³, evitando distorções anatômicas, assegurando consistência volumétrica e melhor aprendizado de padrões 3D pela rede.

4.	Correção de orientação (Orientationd): volumes convertidos para padrão RAS (Right-Anterior-Superior). Isso garante que:
•	O modelo aprenda localização anatômica relativa (direita/esquerda, superior/inferior)
•	Evita confusões em regiões simétricas (como rins, pulmões, hemisférios cerebrais), a correção garante que o modelo não troque esquerda e direita. Para estruturas assimétricas (como fígado ou baço), garante que elas estejam sempre no mesmo lado anatômico, conforme a convenção (fígado à direita, baço à esquerda).

5.	Normalização de intensidade (NormalizeIntensityd): intensidades ajustadas para média zero e desvio padrão unitário. Isso permite que o modelo:
•	Seja robusto a variações de intensidade entre scanners
•	Aprenda padrões anatômicos sem viés de intensidade absoluta

6.	Redimensionamento (ResizeD): volumes ajustados para 128 × 128 × 128 voxels, equilibrando detalhes anatômicos e limitações de memória GPU. Permite treinamento em GPU comum. Isso foi feito para permitir treinamento em GPU comum e reduzir o tempo computacional de inferência.

7.	Conversão final para tensor (EnsureTyped): compatível com MONAI para treinamento e inferência, garantindo integração com DataLoaders, pipelines e funções de loss.

Tabela 2.1: Resumo do Pré-Processamento
<img width="785" height="220" alt="image" src="https://github.com/user-attachments/assets/99cd69a2-2c68-4ed3-a451-640190306b6c" />

 
Após essas transformações, todos os volumes apresentaram shape uniforme, pronto para entrada no modelo.


2.4.
Aumento de Dados (Data Augmentation)

Para aumentar a robustez do modelo e prevenir sobreajuste, são aplicadas transformações geométricas aleatórias:

•	Transformações Espaciais:
o	Rotações em múltiplos eixos
RandRotate90d: Rotações de 90°, 180° ou 270°

o	Espelhamentos (flips) sagitais, coronais e axiais
RandFlipd: Espelhamento aleatório nos três eixos espaciais

o	Translações e escalas aleatórias
RandCropByPosNegLabeld: Recortes aleatórios balanceados (1 positivo:1 negativo)

o	Preenchimento
SpatialPadd: Preenchimento para garantir tamanho mínimo para processamento, preenchimento com zeros (fundo), compatível com arquitetura da rede

•	Amostragem Balanceada:
o	Garante representação equilibrada entre tecido pulmonar e fundo
o	Previne viés em direção à classe majoritária



2.5.
Arquitetura da U-Net 3D

O modelo U-Net 3D foi implementado por meio do módulo monai.networks.nets.UNet da biblioteca MONAI. A definição dos parâmetros arquiteturais considerou a resolução dos volumes de entrada, a capacidade de memória da GPU e a complexidade anatômica da estrutura de interesse.

O modelo foi inicializado com pesos aleatórios e treinado do zero, utilizando monitoramento contínuo da métrica Dice no conjunto de validação. O melhor modelo foi automaticamente salvo com base no maior valor de Dice alcançado, garantindo que o resultado final representasse o estado de treinamento de maior desempenho.

A U-Net 3D implementada segue a estrutura clássica proposta por Ronneberger et al. (2015), com adaptações para o processamento de imagens volumétricas tridimensionais. Suas principais configurações são:
•	Entrada: volume tridimensional (C, D, H, W), com C = 1, representando imagens em escala de cinza;
•	Saída: máscara binária tridimensional com C = 1, correspondente à estrutura segmentada;
•	Filtros por nível do encoder: (8, 16, 32, 64), aumentando progressivamente conforme a profundidade da rede;
•	Strides (downsampling): (2, 2, 2), utilizados para redução sistemática da dimensionalidade espacial;
•	Conexões de salto (skip connections): preservam detalhes anatômicos aprendidos nas camadas iniciais;
•	Unidades residuais: uma por nível, empregadas para evitar a degradação do gradiente e permitir aprendizado mais profundo;
•	Normalização em lote (Batch Normalization) e função de ativação ReLU aplicadas em todas as camadas, a fim de estabilizar o treinamento e acelerar a convergência.


A arquitetura compreende três componentes principais:
1.	Encoder 3D, responsável por capturar padrões locais e globais por meio de convoluções e operações de pooling;
2.	Decoder 3D, encarregado da reconstrução da máscara segmentada por meio de upsampling e convoluções sucessivas;
3.	Conexões de salto (skip connections), que integram informações de alta resolução do encoder com características contextuais do decoder, preservando detalhes espaciais relevantes.

A escolha da U-Net 3D fundamenta-se em três aspectos principais:
•	sua estrutura em formato de “U”, que possibilita o contexto global de cada voxel dentro do volume;
•	o uso de convoluções tridimensionais (3D), que mantêm as relações espaciais entre fatias e capturam a coerência anatômica;
•	e as conexões de salto, essenciais para a segmentação precisa de órgãos pequenos ou estruturas de baixo contraste.
Essa configuração arquitetural permite ao modelo capturar informações volumétricas detalhadas, promovendo alto desempenho e robustez em tarefas de segmentação médica tridimensional.


2.6.
Treinamento do Modelo

O treinamento do modelo U-Net 3D foi conduzido considerando princípios voltados à eficiência do aprendizado e à robustez na generalização dos resultados. O processo contemplou 450 épocas de treinamento, com execução preferencial em GPU (CUDA), explorando a aceleração computacional oferecida pelo processamento paralelo em unidades gráficas. Quando a GPU não estava disponível, o treinamento foi automaticamente realizado em CPU, assegurando compatibilidade e portabilidade do modelo entre diferentes ambientes computacionais.

O pipeline de dados seguiu o pré-processamento descrito na Seção 3.3, no qual cada volume foi padronizado para espaçamento isotrópico, orientação RAS (Right–Anterior–Superior) e escala de intensidade normalizada. Essa uniformização foi fundamental para reduzir discrepâncias entre exames provenientes de diferentes protocolos de aquisição, favorecendo a consistência dos dados e a eficiência do aprendizado da rede.

Devido ao elevado consumo de memória associado ao processamento de volumes tridimensionais, foi adotado batch size igual a 1, permitindo o treinamento com volumes completos sem prejuízo à integridade das informações espaciais.

A função de perda utilizada foi a Dice Loss, amplamente reconhecida como adequada para segmentação médica binária, por penalizar de forma proporcional à sobreposição entre a máscara predita e a máscara de referência (ground truth). Essa característica torna a Dice Loss particularmente eficaz em cenários de desbalanceamento de classes, nos quais a proporção de voxels pertencentes à estrutura de interesse é significativamente menor em relação ao fundo.

Como otimizador, empregou-se o Adam, com taxa de aprendizado adaptativa em seus parâmetros padrão, selecionado por sua estabilidade numérica e capacidade de convergência rápida em arquiteturas profundas.

Durante o treinamento, a métrica Dice foi monitorada a cada época no conjunto de validação. O modelo apresentando o melhor desempenho de validação foi automaticamente salvo (mecanismo de checkpoint), assegurando que o modelo final correspondesse ao estado de aprendizado de maior desempenho obtido durante o processo. O mecanismo de early stopping também foi empregado para interromper o treinamento caso não houvesse melhora significativa na métrica de validação após um número pré-definido de épocas, prevenindo overfitting e otimizando o uso de recursos computacionais.


2.7.
Inferência

A inferência em novos volumes seguiu os mesmos passos de pré-processamento aplicados ao treino.

1.	Carregamento do Volume: Arquivo NRRD lido com a biblioteca nrrd e processado pelos mesmos transforms do treinamento, garantindo consistência.

2.	Adição de dimensão de batch: Tensor com shape (1, 1, D, H, W) para compatibilidade com o modelo 3D.

3.	Sliding Window Inference: Necessário para processar volumes grandes que não cabem na memória GPU de uma só vez.
    o	Tamanho da janela: (96, 96, 96).
    o	Sobreposição entre janelas: 25%.
    o	Batch size: 1.

4.	Aplicação de sigmoid: Transformou logits da saída em probabilidades entre 0 e 1.

5.	Threshold de 0.5: Para binarização da máscara predita.

6.	Resultados:
•	Shape da máscara predita: (128, 128, 128).
•	Percentual de voxels positivos: Depende do volume, mas o pipeline permite análise quantitativa do volume segmentado.
•	Dice Score médio no conjunto de teste: 0.7608. 
•	Desvio padrão do Dice: 0.0959.




### 3. Resultados

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin pulvinar nisl vestibulum tortor fringilla, eget imperdiet neque condimentum. Proin vitae augue in nulla vehicula porttitor sit amet quis sapien. Nam rutrum mollis ligula, et semper justo maximus accumsan. Integer scelerisque egestas arcu, ac laoreet odio aliquet at. Sed sed bibendum dolor. Vestibulum commodo sodales erat, ut placerat nulla vulputate eu. In hac habitasse platea dictumst. Cras interdum bibendum sapien a vehicula.

Proin feugiat nulla sem. Phasellus consequat tellus a ex aliquet, quis convallis turpis blandit. Quisque auctor condimentum justo vitae pulvinar. Donec in dictum purus. Vivamus vitae aliquam ligula, at suscipit ipsum. Quisque in dolor auctor tortor facilisis maximus. Donec dapibus leo sed tincidunt aliquam.

### 4. Conclusões

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin pulvinar nisl vestibulum tortor fringilla, eget imperdiet neque condimentum. Proin vitae augue in nulla vehicula porttitor sit amet quis sapien. Nam rutrum mollis ligula, et semper justo maximus accumsan. Integer scelerisque egestas arcu, ac laoreet odio aliquet at. Sed sed bibendum dolor. Vestibulum commodo sodales erat, ut placerat nulla vulputate eu. In hac habitasse platea dictumst. Cras interdum bibendum sapien a vehicula.

Proin feugiat nulla sem. Phasellus consequat tellus a ex aliquet, quis convallis turpis blandit. Quisque auctor condimentum justo vitae pulvinar. Donec in dictum purus. Vivamus vitae aliquam ligula, at suscipit ipsum. Quisque in dolor auctor tortor facilisis maximus. Donec dapibus leo sed tincidunt aliquam.

---

Matrícula: 123.456.789

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*







