<!-- antes  de enviar a versão final, solicitamos que todos os comentários, colocados para orientação ao aluno, sejam removidos do arquivo -->
# Nome do projeto

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

Nos últimos anos, a análise de exames de imagens tridimensionais tornou-se um componente fundamental na prática clínica e na pesquisa biomédica. Exames como tomografia computadorizada (TC) e ressonância magnética (RM) geram volumes ricos em informação, que permitem avaliar órgãos, tecidos e estruturas complexas com alta precisão. Entretanto, a segmentação manual dessas imagens é uma tarefa extremamente demorada e pode apresentar inconsistências devido à qualidade variável das imagens. O advento das redes neurais convolucionais (CNNs) e, especificamente, da arquitetura U-Net 3D, revolucionou o campo de segmentação de imagens médicas, permitindo a análise direta em volumes completos enquanto preserva a continuidade anatômica das estruturas.
Este trabalho apresenta o desenvolvimento e avaliação de um sistema automatizado para segmentação pulmonar fetal em imagens de ressonância magnética 3D, utilizando a arquitetura U-Net 3D implementada no framework MONAI. O pipeline abrange desde o carregamento de dados NRRD até a geração de máscaras segmentadas, incluindo pré-processamento, normalização, reamostragem para voxel isotrópico de 1,5 mm³, redimensionamento para 128³ voxels e aumento de dados.
O modelo foi treinado em GPU NVIDIA com monitoramento do coeficiente de Dice e função de perda combinada (Dice + Cross Entropy), alcançando 0,7608 no conjunto de teste — resultado considerado clinicamente relevante. O uso da técnica de janela deslizante (sliding window) permitiu processar volumes completos mantendo consistência espacial.
Apesar do bom desempenho, a necessidade de reduzir a resolução das imagens para o treinamento limitou a fidelidade das máscaras e causou perda de detalhes na tentativa de reescalar as máscaras. Mesmo assim, foi possível obter estimativas volumétricas úteis aplicando fator de escala. Como trabalho futuro, planeja-se treinar o modelo em resolução original com hardware mais potente, a fim de aumentar a precisão, utilização das máscaras e preservar a riqueza de detalhes para aplicações clínicas e de pesquisa.

Palavras-chave
Segmentação Automática, Ressonância Magnética Fetal, U-Net 3D, MONAI, Deep Learning, Processamento de Imagens Médicas, Pulmão Fetal.

### Abstract <!-- Opcional! Caso não aplicável, remover esta seção -->

<!-- trocar o texto abaixo pelo resumo do trabalho, em inglês -->

In recent years, the analysis of three-dimensional imaging exams has become a fundamental component in clinical practice and biomedical research. Exams such as computed tomography (CT) and magnetic resonance imaging (MRI) generate information-rich volumes that allow for the evaluation of organs, tissues, and complex structures with high precision. However, the manual segmentation of these images is an extremely time-consuming task and can present inconsistencies due to the variable quality of the images. The advent of convolutional neural networks (CNNs) and, specifically, the U-Net 3D architecture, has revolutionized the field of medical image segmentation, allowing direct analysis of entire volumes while preserving the anatomical continuity of structures.
This work presents the development and evaluation of an automated system for fetal lung segmentation in 3D magnetic resonance imaging, using the U-Net 3D architecture implemented in the MONAI framework. The pipeline covers everything from loading NRRD data to generating segmented masks, including preprocessing, normalization, resampling to 1.5 mm³ isotropic voxels, resizing to 128³ voxels, and data augmentation.
The model was trained on an NVIDIA GPU with monitoring of the Dice coefficient and combined loss function (Dice + Cross Entropy), achieving 0.7608 in the test set — a result considered clinically relevant. The use of the sliding window technique allowed processing entire volumes while maintaining spatial consistency.
Despite the good performance, the need to reduce the image resolution for training limited the fidelity of the masks and caused a loss of detail when attempting to rescale them. Even so, it was possible to obtain useful volumetric estimates by applying scaling factor. As future work, it is planned to train the model at its original resolution with more powerful hardware in order to increase accuracy, mask utilization, and preserve the richness of detail for clinical and research applications.

Keywords
Automatic Segmentation, Fetal Magnetic Resonance Imaging, U-Net 3D, MONAI, Deep Learning, Medical Image Processing, Fetal Lung.


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
