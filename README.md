Controle de nível de caldeira aquatubular com PID e lógica fuzzy

Este repositório reúne o código-fonte desenvolvido para as simulações apresentadas no Trabalho de Conclusão de Curso “Aplicação de controle fuzzy na estabilização do nível em caldeiras frente aos efeitos shrink e swell”.

O código implementa:

a representação linearizada em espaço de estados de uma caldeira aquatubular;
a incorporação de uma aproximação empírica dos efeitos transitórios shrink e swell;
um controlador proporcional-integral-derivativo (PID);
um controlador baseado em lógica fuzzy;
a comparação quantitativa entre os controladores por meio das métricas IAE, ISE, ITAE, máximo desvio percentual e último instante fora da faixa de tolerância de ±2%;
a geração automática dos gráficos empregados no trabalho.
Arquivo principal

tcc_controle_caldeira_aquatubular_v15.py

Ambiente de execução

O código foi desenvolvido em Python e utiliza as bibliotecas NumPy, SciPy e Matplotlib.

Autor

Lucas Laporti Pinto
Curso de Engenharia de Controle e Automação
Instituto Federal do Espírito Santo — Campus Linhares
