# DEVLOG — Atividade Ponderada M7

Este documento registra o processo de desenvolvimento da solução, mostrando como pensei a arquitetura, quais decisões eu tomei e como cada etapa foi validada na prática. A ideia foi manter uma narrativa fiel ao desenvolvimento real, sem tornar a documentação apenas uma lista de comandos. Aqui, eu descrevo não só o que foi feito, mas também por que foi feito assim.

Durante a implementação, usei o Claude como apoio para revisar trechos de código, sugerir ajustes de organização e debugar alguns pontos específicos da lógica. Mesmo assim, as escolhas centrais da solução — moeda, horizonte, modelo, arquitetura, forma de armazenar o artefato e a API — foram minhas. A ferramenta me ajudou a acelerar a escrita e a organização, mas a direção técnica e a justificativa da solução vieram do meu entendimento do problema.

## Resumo da solução
A solução foi pensada como um projeto em duas etapas bem separadas. Primeiro, o container `trainer` realiza a preparação dos dados, treina um modelo para prever o fechamento do dia seguinte do BTC-USD e salva o artefato em um volume compartilhado. Depois, o container `backend` lê esse artefato e expõe uma API em FastAPI para responder às predições.

A ideia era manter o treinamento fora do backend para que o serviço de inferência ficasse mais leve, previsível e fácil de testar. O modelo treinado é salvo em `./models` para ser consumido pelo backend em uma segunda etapa, o que também deixa claro a lógica de entrega do artefato pedida no enunciado.

## Fase 0 — Planejamento e desenho da arquitetura
No início, eu defini que o problema seria prever o fechamento do dia seguinte do BTC-USD usando dados históricos diários. Essa escolha foi simples e funcional: a instituição pede um problema de previsão em série temporal, mas não exige uma solução financeira complexa. O objetivo era demonstrar integração entre treinamento, artefato do modelo e inferência em containers, sem perder foco na arquitetura e no processo de desenvolvimento.

A decisão de usar dados diários de fechamento também foi importante porque o dado tem baixa complexidade e é fácil de obter. O valor de fechamento do ativo é um indicador clássico em previsão de séries temporais e ajuda a manter o projeto compreensível. A minha escolha foi manter a solução centrada em um dado simples e bem definido, em vez de complicar o problema com múltiplas fontes e indicadores.

A arquitetura ficou em Docker Compose com dois serviços principais: `trainer` e `backend`. O treinamento cria o artefato e salva em volume compartilhado; o backend lê esse artefato em modo somente leitura. Essa decisão foi essencial para demonstrar a transferência do modelo entre os componentes, como exigido no enunciado.

O diagrama UML foi desenhado para representar exatamente esse fluxo. Ele mostra a origem dos dados, o processo de treinamento, o artefato gerado e a chamada feita ao backend para predição.

![Diagrama UML da arquitetura](docs/arquitetura.png)

Esse diagrama foi importante porque ele permite que outra pessoa entenda, de maneira visual, como os elementos se conectam no projeto. O desenho também explica como o modelo treinado chega ao container de inferência: por meio do volume compartilhado `./models`.

## Fase 1 — Dados, features e escolha do modelo
A parte mais importante para mim foi transformar os dados em uma estrutura útil para o modelo. Em vez de usar o preço absoluto do BTC diretamente em cada instante, eu optei por usar relações entre preços em uma janela de 7 dias. Essa escolha foi feita porque o valor absoluto do ativo muda bastante ao longo do tempo, e esse tipo de escala pode distorcer a relação entre os dados. Ao usar razões como proporções entre preços consecutivos e dentro da janela, a representação fica mais estável e mais adequada para uma regressão linear.

Também foi uma decisão consciente usar um split temporal, em vez de embaralhar os dados. Em problemas de séries temporais, essa é a abordagem correta, porque o modelo precisa ser testado em um cenário futuro e realista. Isso evita validar o modelo com dados que vieram de um momento que não corresponde ao cenário real de operação.

Sobre o modelo, eu escolhi o Ridge. A escolha foi minha porque ele é simples, interpretável, eficiente e facilmente serializável em um artefato `.joblib`. Eu poderia ter escolhido algo mais sofisticado, mas a atividade não exige um modelo financeiro de alto nível; ela pede uma solução funcional, bem documentada e com integração correta em containers. O Ridge atende muito bem a esse objetivo, principalmente por ser uma escolha simples e clara para um modelo linear regularizado.

Ainda assim, eu também comparei o modelo com um baseline de “amanhã = hoje”, porque isso me permitiu testar se a predição realmente agrega algum valor ou se simplesmente repete a tendência mais básica do próprio preço. Esse ponto foi muito importante para a minha análise final, porque a natureza do mercado cripto é altamente ruidosa e nem sempre vale a pena criar uma solução complexa quando o problema tem um sinal fraco.

## Fase 2 — Treinamento e exportação do artefato
O treinamento foi implementado em `training/train.py`. O fluxo ficou assim:

1. verificar se o CSV existe em `data/btc_usd_daily.csv`;
2. se não existir, baixar os dados históricos via `yfinance`;
3. limpar e ordenar os dados por data;
4. construir janelas de 7 dias;
5. gerar as features com a lógica compartilhada em `common/features.py`;
6. separar treino e teste de forma cronológica;
7. treinar o Ridge;
8. calcular métricas no conjunto de teste;
9. salvar o modelo e o arquivo de métricas em `./models`.

Os comandos executados foram:

```bash
docker compose build trainer
docker compose run --rm trainer
```

O resultado observado do treinamento confirmava a estrutura do dataset e as métricas finais. O arquivo CSV usado tinha 3200 linhas, com dados desde 2018-01-01 até 2026-10-05.

A execução do treino gerou os arquivos do modelo no volume compartilhado. As métricas efetivas foram:

- n_train: 2554
- n_test: 639
- model_mae: 1404.4327401806067
- model_rmse: 1982.996473171243
- baseline_mae: 1397.2352430555557
- baseline_rmse: 1977.067026993581

Em outras palavras, o modelo ficou muito próximo do baseline. Isso faz sentido dado o problema de predição financeira e a natureza altamente incerta do mercado de criptomoedas. A análise foi honesta: o modelo não foi claramente superior ao baseline, mas ele demonstrou a pipeline correta, o uso do split temporal e a geração do artefato em um formato pronto para inferência.

## Fase 3 — Backend de inferência em FastAPI
O backend foi implementado em `backend/app.py`. Ele foi pensado para carregar o modelo salvo no volume compartilhado e expor endpoints simples para verificação e inferência. A estrutura do backend foi pensada para seguir a exigência do enunciado: o backend precisa ser em Python, usar o modelo treinado e dar uma resposta útil para a aplicação cliente.

Os endpoints implementados foram:

- `GET /health`: verifica se o serviço está vivo e se o modelo foi carregado;
- `GET /model-info`: retorna o campo da janela, a versão do scikit-learn e as métricas;
- `POST /predict`: recebe uma lista de preços e devolve o próximo fechamento estimado.

A validação foi feita de forma clara para impedir entradas inválidas. Eu também mantive a lógica de features em um módulo compartilhado para garantir que treino e inferência utilizem a mesma transformação de dados. Isso reduz o risco de inconsistência entre os dois containers.

Os comandos usados para subir o backend foram:

```bash
docker compose up --build -d
docker compose ps
docker compose logs backend
```

A execução real confirmou que o backend ficou funcional. O estado do serviço ficou como esperado:

```json
{"status":"ok","model_loaded":true}
```

Isso foi essencial para validar que a fase de treinamento e a fase de inferência estavam realmente integradas.

## Fase 4 — Testes do fluxo completo
Depois de subir os componentes, eu validei o fluxo real da aplicação e o comportamento do backend em operação. Os testes executados foram:

```bash
curl localhost:8000/health
curl localhost:8000/model-info
bash client/request_example.sh
```

E os resultados observados foram:

```json
{"status":"ok","model_loaded":true}
```

```json
{"window":7,"sklearn_version":"1.5.2","metrics":{"n_train":2554,"n_test":639,"model_mae":1404.4327401806067,"model_rmse":1982.996473171243,"baseline_mae":1397.2352430555557,"baseline_rmse":1977.067026993581}}
```

```json
{"last_close":62300.0,"predicted_next_close":62398.91,"disclaimer":"Predição experimental; não é recomendação de investimento."}
```

Esse teste foi importante porque comprovou que o backend, de fato, foi capaz de carregar o artefato treinado e responder uma predição real. Além disso, eu também validei o caso inválido do payload para confirmar que a API rejeitava entradas insuficientes e respondia com erro apropriado.

## Fase 5 — Dificuldades e como foram resolvidas
A principal dificuldade foi equilibrar a simplicidade da solução com a necessidade de representar corretamente a arquitetura e o fluxo do modelo. Como o mercado financeiro é muito ruidoso, há a tentação de complicar a solução com muitos indicadores e modelos mais sofisticados. Eu escolhi manter uma abordagem clara e robusta, porque a atividade valoriza a integração, a reprodução e a documentação, e não apenas o desempenho extremo do modelo.

Outra dificuldade foi garantir que o artefato gerado fosse compatível com o backend. Isso exigiu atenção às versões das bibliotecas e à forma como o modelo era salvo e lido entre os contêineres. O cuidado com a consistência entre o ambiente de treinamento e o ambiente de inferência foi importante para evitar problemas de execução.

Também houve atenção ao fato de que a lógica de features precisava ser idêntica nos dois lados. Para isso, eu centralizei a geração de features em um módulo compartilhado, evitando esse tipo de divergência.

## Como salvar as evidências no repositório
Para atender ao enunciado, o ideal é salvar as evidências em `docs/evidencias` com nomes claros e fáceis de identificar. O que eu recomendo é:

1. fazer o print do diagrama UML e salvar como `docs/evidencias/01-diagrama-uml.png`;
2. fazer o print do treinamento e salvar como `docs/evidencias/02-treino.png`;
3. fazer o print do `docker compose ps` ou do `docker compose logs backend` e salvar como `docs/evidencias/03-compose.png`;
4. fazer o print da resposta do endpoint de predição e salvar como `docs/evidencias/04-predict.png`;
5. manter tudo comitado no GitHub junto com o projeto.

Os comandos para gerar essas emissões podem ser os seguintes:

```bash
docker compose up --build -d
curl localhost:8000/health
curl localhost:8000/model-info
bash client/request_example.sh
```

Depois disso, basta selecionar a saída do terminal ou a tela do navegador e salvar em `docs/evidencias/` dentro do repositório. Isso deixa a pasta de evidências pronta para a correção do professor.

## Limitações conhecidas
- O modelo usa apenas preço histórico, sem outros indicadores econômicos ou de mercado.
- O problema é intrinsecamente ruidoso, e o modelo tende a ficar próximo do baseline.
- A solução não faz retreino automático nem monitoramento em produção.
- A validação foi feita em um único split temporal, o que é apropriado para a atividade, mas não é a forma mais completa de avaliação financeira.

## Conclusão
A solução foi concluída de forma funcional e coerente com o enunciado. O treinamento gerou o artefato do modelo, o backend carregou esse artefato corretamente e a predição foi respondida em operação real. Em termos de arquitetura, produtividade e organização, a solução atende ao que a atividade pede.

A parte mais importante foi a decisão de manter a solução simples, reprodutível e explicável. Isso foi uma escolha consciente, porque a atividade valoriza a integração dos componentes e a documentação do processo, não apenas a precisão do modelo. Se eu fosse continuar a solução, o próximo passo seria testar outras abordagens de validação, como walk-forward, e explorar mais features sem perder a clareza da arquitetura que foi implementada.
