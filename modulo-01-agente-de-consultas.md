# Módulo 1: Agente de Consultas (clima, câmbio, CEP)

**Nível:** 1 · **Duração:** 2 semanas · **Orçamento:** < US$ 1
**Dificuldade nova dominante:** escrever o loop agentic **à mão**, sem framework.

> **Regra inegociável deste módulo:** zero frameworks de agente. Nada de LangChain, LangGraph, CrewAI, Pydantic AI, Agents SDK. Só o SDK oficial do modelo, `httpx` (ou `requests`) e Pydantic.
>
> Motivo: a recomendação explícita da Anthropic em "Building effective agents" é que frameworks escondem prompts e respostas. Este é o único módulo da trilha em que você vê o mecanismo cru. Se você pular isso, nos módulos 5 e 7 vai estar depurando abstrações sem saber o que elas abstraem.

---

## 1. Diagnóstico de abertura

Não há módulo anterior, então em vez de quiz de recuperação este módulo abre com um **baseline**. Responda por escrito, sem pesquisar, antes de ler o resto:

1. Quando um LLM "chama uma ferramenta", o que exatamente ele devolve na resposta da API?
2. Quem executa a função: o modelo ou o seu código?
3. O que acontece com o resultado da função depois que ela roda?
4. Quantas vezes o modelo é chamado numa pergunta que precisa de duas ferramentas em sequência?
5. O que impede um agente de entrar em loop infinito?

Guarde as respostas num arquivo `baseline.md`. No fim do módulo você vai reler. Serve como medida direta do que o módulo te ensinou, e é a primeira medição da sua calibração (§4 do Documento 1).

---

## 2. O problema

**O agente:** responde perguntas de usuário combinando três ferramentas externas:

- Previsão do tempo para uma cidade
- Cotação de moeda
- Consulta de endereço por CEP

**Exemplos de pergunta que ele deve resolver:**

- "Vai chover em Campinas amanhã?" → 1 ferramenta
- "Quanto custa 100 dólares hoje?" → 1 ferramenta
- "Qual a cidade do CEP 13140-000 e como vai estar o tempo lá?" → 2 ferramentas **em sequência** (a segunda depende do resultado da primeira)
- "Quem descobriu o Brasil?" → **nenhuma ferramenta**, o agente deve responder sem inventar chamada
- "Qual a cotação do Bitcoin em 1998?" → nenhuma ferramenta serve, o agente deve **recusar**

**Sem:** estado entre conversas, memória de longo prazo, banco de dados, interface web. Uma função Python que recebe uma pergunta e devolve uma resposta. Ponto.

**Por que este é o módulo 1:** dos oito eixos de dificuldade (§3.1 do Documento 0), este agente só exercita o primeiro (número de ferramentas) e, de leve, o sexto (avaliação, e aqui as respostas são verificáveis). Sem estado, sem assincronia, sem integração autenticada, sem superfície de segurança relevante. É o ambiente mais limpo possível para ver o loop funcionando.

---

## 3. Conceitos novos

### 3.1 O que realmente acontece numa chamada de ferramenta

Antes de qualquer código, o modelo mental. Isto é o que mais gente erra:

> **O modelo nunca executa nada.** Ele devolve um pedaço de JSON dizendo "eu gostaria de chamar a função X com estes argumentos". Quem executa é o **seu código**. Depois você devolve o resultado para o modelo numa nova mensagem, e ele continua.

Um turno com ferramenta tem **no mínimo duas chamadas à API**:

```
Chamada 1:  você → modelo:  [mensagens do usuário] + [definições de ferramentas]
            modelo → você:  stop_reason="tool_use", quer chamar get_clima(cidade="Campinas")

            (seu código executa get_clima e obtém o resultado)

Chamada 2:  você → modelo:  [histórico] + [resultado da ferramenta]
            modelo → você:  stop_reason="end_turn", texto final para o usuário
```

Se a pergunta exigir duas ferramentas em sequência, são **três** chamadas à API. É por isso que agente custa mais caro que chat: cada passo reenvia o histórico inteiro.

### 3.2 Nível 1 — Demonstração isolada: definindo uma ferramenta

> Domínio proposital: **biblioteca**. Não é o do seu agente.

Uma ferramenta é só um dicionário com nome, descrição e um JSON Schema dos parâmetros.

```python
# COMO FUNCIONA: a estrutura de uma definição de ferramenta
# Domínio: biblioteca. Não use isto no seu agente.

FERRAMENTA_BUSCAR_LIVRO = {
    "name": "buscar_livro",
    "description": (
        "Busca um livro no acervo da biblioteca pelo título ou pelo autor. "
        "Use quando o usuário perguntar sobre disponibilidade, localização "
        "na estante ou data de devolução de um livro específico. "
        "NÃO use para recomendações gerais de leitura."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "termo": {
                "type": "string",
                "description": "Título ou nome do autor a buscar.",
            },
            "campo": {
                "type": "string",
                "enum": ["titulo", "autor"],
                "description": "Em qual campo buscar o termo.",
            },
        },
        "required": ["termo", "campo"],
    },
}
```

**Três coisas para notar, e elas importam mais do que parecem:**

1. **A `description` é prompt.** É o texto que o modelo lê para decidir se chama a ferramenta. Dizer *quando usar* e *quando não usar* vale mais que descrever o que ela faz.
2. **`enum` elimina uma classe inteira de erro.** Com `enum`, o modelo não pode inventar um valor de `campo`. Sem ele, você vai receber `"titulo "`, `"título"`, `"TITLE"` e vai tratar isso no código.
3. **`required` é levado a sério.** Campo opcional é campo que às vezes vem vazio e você tem que tratar.

> **Dado relevante:** no agente de SWE-bench, a Anthropic relata ter gasto **mais tempo otimizando as ferramentas do que o prompt**. A intuição comum é o contrário. Quando o seu agente errar neste módulo, suspeite primeiro da descrição da ferramenta, não do system prompt.

### 3.3 Nível 1 — Demonstração isolada: gerando o schema com Pydantic

Escrever JSON Schema à mão cansa e desincroniza com o código. Pydantic gera o schema a partir do tipo.

```python
# COMO FUNCIONA: Pydantic gerando JSON Schema
# Domínio: pizzaria. Não é o do seu agente.

from pydantic import BaseModel, Field
from typing import Literal

class ArgsPedido(BaseModel):
    sabor: str = Field(description="Sabor da pizza, ex: calabresa")
    tamanho: Literal["P", "M", "G"] = Field(description="Tamanho da pizza")
    adicionais: list[str] = Field(
        default_factory=list,
        description="Ingredientes extras. Lista vazia se não houver.",
    )

# Isto produz o dicionário que vai no campo input_schema:
print(ArgsPedido.model_json_schema())

# E, na volta, valida o que o modelo mandou:
args_do_modelo = {"sabor": "calabresa", "tamanho": "G"}
pedido = ArgsPedido(**args_do_modelo)   # ValidationError se vier errado
print(pedido.tamanho)
```

`Literal["P","M","G"]` vira `enum` no schema automaticamente. Você escreve o tipo uma vez e ganha schema **e** validação.

> **Ponto importante:** validar na volta não é paranoia. O modelo erra argumento, principalmente quando a descrição é ambígua. Validar transforma um bug silencioso (campo errado que vira `None` lá na frente) num erro explícito que você pode devolver ao modelo para ele corrigir.

### 3.4 Nível 1 — Demonstração isolada: o formato da resposta com `tool_use`

```python
# COMO FUNCIONA: lendo uma resposta que pede ferramenta
# Domínio: academia. Não é o do seu agente.

resposta = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=[FERRAMENTA_CONSULTAR_TREINO],
    messages=[{"role": "user", "content": "Qual meu treino de hoje?"}],
)

print(resposta.stop_reason)   # "tool_use"

# O content é uma LISTA de blocos. Pode ter texto E pedidos de ferramenta.
for bloco in resposta.content:
    if bloco.type == "text":
        print("O modelo falou:", bloco.text)
    elif bloco.type == "tool_use":
        print("ID da chamada:", bloco.id)      # você precisa guardar este id
        print("Ferramenta:", bloco.name)
        print("Argumentos:", bloco.input)      # já vem como dict
```

**O `bloco.id` é o detalhe que mais gente ignora.** Quando você devolve o resultado, precisa referenciar exatamente esse id. Se o modelo pediu três ferramentas em paralelo, são três ids e três resultados, e trocar um pelo outro produz um bug difícil de enxergar: o agente responde com confiança usando o dado errado.

### 3.5 Nível 1 — Demonstração isolada: devolvendo o resultado

```python
# COMO FUNCIONA: a mensagem de retorno de ferramenta
# Domínio: academia. Não é o do seu agente.

# 1. Você anexa a resposta do assistente ao histórico, inteira:
mensagens.append({"role": "assistant", "content": resposta.content})

# 2. Você executa a função de verdade e monta o retorno:
resultado = consultar_treino(**bloco.input)

mensagens.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": bloco.id,          # o mesmo id de antes
        "content": str(resultado),        # texto. Serialize dict/objeto
    }],
})

# 3. Você chama o modelo de novo, com o histórico crescido.
```

Três armadilhas aqui:

- O retorno vai com `role: "user"`, não `"assistant"`. Parece errado e é assim mesmo.
- `content` do `tool_result` tem que ser **texto**. Dicionário vai como JSON serializado.
- Você precisa anexar a resposta **completa** do assistente (`resposta.content`), não só o bloco de ferramenta. Se omitir o texto que veio junto, o modelo perde o próprio raciocínio.

### 3.6 Nível 1 — Demonstração isolada: erro de ferramenta

Ferramenta externa falha. API fora do ar, timeout, parâmetro inválido, recurso inexistente. A reação instintiva é deixar a exceção subir e quebrar o agente. Está errado.

```python
# COMO FUNCIONA: erro de ferramenta vira observação, não exceção
# Domínio: biblioteca. Não é o do seu agente.

def executar_com_seguranca(func, args: dict) -> tuple[str, bool]:
    """Devolve (texto_do_resultado, houve_erro)."""
    try:
        return str(func(**args)), False
    except ValueError as e:
        # Erro que o MODELO pode corrigir sozinho (argumento ruim).
        return f"Erro de argumento: {e}. Verifique os parâmetros e tente de novo.", True
    except TimeoutError:
        # Erro que o modelo NÃO pode corrigir. Ele precisa saber para desistir.
        return "A busca no acervo não respondeu a tempo. Serviço indisponível.", True

# E o retorno ao modelo marca o erro explicitamente:
texto, houve_erro = executar_com_seguranca(buscar_livro, bloco.input)

mensagens.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": bloco.id,
        "content": texto,
        "is_error": houve_erro,
    }],
})
```

**O princípio:** erro de ferramenta é **observação do ambiente**, não falha do programa. O agente precisa *ver* o erro para poder reagir, seja corrigindo o argumento, seja dizendo ao usuário que o serviço caiu.

A distinção entre erro corrigível e não corrigível importa. Se o agente não souber que o timeout é irrecuperável, ele tenta de novo, de novo, e você descobre no fim do mês.

### 3.7 Nível 1 — Demonstração isolada: limite de passos

```python
# COMO FUNCIONA: por que todo loop precisa de teto
# Domínio: pizzaria. Não é o do seu agente.

MAX_PASSOS = 8

for passo in range(MAX_PASSOS):
    resposta = chamar_modelo(mensagens)
    if resposta.stop_reason != "tool_use":
        break
    processar_ferramentas(resposta, mensagens)
else:
    # Este else pertence ao for e roda se o loop terminou SEM break.
    raise LimiteDePassosExcedido(
        f"O agente não concluiu em {MAX_PASSOS} passos."
    )
```

Dois agentes entram em loop pelos mesmos motivos: ferramenta que devolve sempre erro e o modelo insiste, ou duas ferramentas cujas descrições se sobrepõem e o modelo alterna entre elas.

> Existe um catálogo de 2026 com 63 incidentes de produção envolvendo loops descontrolados e estouro de orçamento. Limite de passos não é zelo excessivo; é o freio de mão.

O `for/else` do Python é o jeito idiomático e quase ninguém usa. O `else` só roda se o `for` terminou sem `break`, que é exatamente a condição "estourei os passos sem concluir".

### 3.8 Nível 1 — Demonstração isolada: resposta estruturada final

Seu agente vai ser avaliado automaticamente. Comparar texto livre com gabarito é ruim. Peça saída estruturada no fim.

```python
# COMO FUNCIONA: forçando saída estruturada via ferramenta terminal
# Domínio: biblioteca. Não é o do seu agente.

from pydantic import BaseModel
from typing import Literal

class RespostaFinal(BaseModel):
    status: Literal["respondido", "sem_dados", "fora_de_escopo"]
    resposta: str
    fontes_usadas: list[str]   # nomes das ferramentas efetivamente chamadas

# Você expõe isto como uma ferramenta a mais, e instrui o modelo
# a SEMPRE terminar chamando-a. O loop para quando ela é chamada.
FERRAMENTA_RESPONDER = {
    "name": "responder",
    "description": (
        "Chame esta ferramenta para entregar a resposta final ao usuário. "
        "Sempre termine por aqui. Use status='fora_de_escopo' quando "
        "nenhuma ferramenta disponível puder responder."
    ),
    "input_schema": RespostaFinal.model_json_schema(),
}
```

Esse padrão (uma "ferramenta terminal") resolve dois problemas de uma vez: dá estrutura para o eval e cria um ponto único e explícito de parada do loop.

### 3.9 Nível 2 — Esqueleto com lacunas

Agora a estrutura do que você vai escrever. **Os `# TODO` são os pontos de decisão. São eles o exercício.**

```python
# ESQUELETO. Complete os TODO. Isto não é o código do seu agente,
# é a forma do problema.

from dataclasses import dataclass

@dataclass
class Ferramenta:
    nome: str
    descricao: str
    schema: dict
    funcao: callable
    # TODO: você precisa de mais algum campo aqui?
    #       Pense: como o loop sabe que ESTA é a ferramenta terminal?
    #       E se uma ferramenta for cara ou lenta, como o loop saberia?


class Agente:
    def __init__(self, ferramentas: list[Ferramenta], max_passos: int):
        self.ferramentas = {f.nome: f for f in ferramentas}
        self.max_passos = max_passos
        # TODO: onde você vai acumular as métricas do DoD
        #       (tokens de entrada, de saída, nº de passos, latência)?
        #       Decida isto AGORA, não depois. O módulo 2 vai depender.

    def _system_prompt(self) -> str:
        # TODO: o que precisa estar aqui que NÃO cabe na descrição
        #       de nenhuma ferramenta individual?
        # TODO: como você instrui o comportamento de recusa
        #       ("quando nenhuma ferramenta serve")?
        ...

    def _executar_ferramenta(self, nome: str, args: dict) -> tuple[str, bool]:
        # TODO: valide args contra o schema ANTES de chamar a função.
        #       O que você devolve ao modelo quando a validação falha?
        # TODO: distinga erro corrigível de erro irrecuperável.
        ...

    def responder(self, pergunta: str) -> RespostaFinal:
        mensagens = [{"role": "user", "content": pergunta}]

        for passo in range(self.max_passos):
            resposta = self._chamar_modelo(mensagens)

            # TODO: acumule as métricas desta chamada aqui.

            # TODO: o modelo pode pedir VÁRIAS ferramentas num único turno
            #       (chamadas paralelas). Seu loop trata isso?
            #       O que acontece se você processar só a primeira?

            # TODO: como você detecta que a ferramenta terminal foi chamada?

            ...

        else:
            # TODO: estourou o limite. O que você devolve?
            #       Exceção? Uma RespostaFinal com status especial?
            #       Qual das duas facilita o eval?
            ...
```

**Se você só completar os TODO mecanicamente, perdeu o módulo.** Cada um tem uma decisão de projeto embutida. Escreva num arquivo `decisoes.md` por que escolheu cada caminho. No módulo 5, quando precisar de checkpointing, você vai reler isso.

---

## 4. Leitura obrigatória

**Antes de escrever uma linha:**

- **"Building effective agents"**, Anthropic (19/12/2024), integral.
  https://www.anthropic.com/engineering/building-effective-agents

Leia de ponta a ponta, não em diagonal. Depois responda por escrito:

1. Pela definição do texto, o agente deste módulo é um agente ou um workflow? Justifique.
2. Quais dos cinco padrões (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) este agente usa? Quais não usa, e por quê?
3. O texto diz para não começar com framework. Qual é o argumento concreto, e você concorda?

**Leitura de referência, conforme a necessidade:**
- Documentação oficial de tool use do SDK que você escolher
- Unit 1 do curso da Hugging Face, seção do loop Thought → Action → Observation

---

## 5. Especificação do agente (sem código)

### 5.1 Comportamento exigido

| Situação | Comportamento esperado |
|---|---|
| Pergunta resolvível com 1 ferramenta | Chama a ferramenta, responde com o dado, `status="respondido"` |
| Pergunta que exige 2 ferramentas em sequência | Chama a primeira, usa o resultado como entrada da segunda, responde |
| Pergunta que exige 2 ferramentas independentes | Pode chamar em paralelo no mesmo turno |
| Pergunta de conhecimento geral | Responde direto, **sem chamar ferramenta**, `status="respondido"` |
| Pergunta fora do alcance das ferramentas | **Recusa explicitamente**, `status="fora_de_escopo"`. Não inventa |
| Ferramenta devolve erro corrigível | Corrige o argumento e tenta de novo (no máximo 1 vez) |
| Ferramenta devolve erro irrecuperável | Informa ao usuário, `status="sem_dados"`. **Não tenta de novo** |
| Cidade/CEP inexistente | `status="sem_dados"`, não inventa um dado plausível |
| Limite de passos atingido | Retorno controlado, nunca travamento |

### 5.2 Restrições técnicas

- **Zero frameworks de agente.** SDK oficial + `httpx`/`requests` + Pydantic.
- **Três ferramentas de domínio + uma terminal.**
- **`MAX_PASSOS` configurável**, default 8.
- **Toda chamada de ferramenta validada** contra o schema antes de executar.
- **Métricas acumuladas por execução:** tokens de entrada, tokens de saída, número de passos, latência total, lista de ferramentas chamadas.
- **Modelo configurável por variável**, não hardcoded. Você vai trocar de modelo nos evals.

```python
# Faça assim desde o primeiro dia. No módulo 2 isto vira roteamento de modelo.
MODEL_RAPIDO = os.environ["MODEL_RAPIDO"]   # barato, para evals
MODEL_PADRAO = os.environ["MODEL_PADRAO"]   # o do agente
```

> **Confira os preços na página oficial do provedor antes de escolher.** Os números do Documento 0 vêm de agregadores e os nomes de modelo de 2026 não foram confirmados nas fontes primárias.

### 5.3 APIs sugeridas (todas gratuitas, sem cartão)

- **Clima:** Open-Meteo (não exige chave)
- **Câmbio:** AwesomeAPI ou exchangerate.host
- **CEP:** ViaCEP ou BrasilAPI

Se alguma estiver fora do ar quando você começar, troque. A escolha da API não é o ponto do módulo.

### 5.4 Estrutura de arquivos sugerida

```
modulo-01/
├── agente/
│   ├── __init__.py
│   ├── loop.py          # o loop agentic
│   ├── ferramentas.py   # definições + implementações
│   └── modelos.py       # schemas Pydantic
├── evals/
│   ├── casos.yaml       # os 30+ casos de teste
│   └── rodar.py         # executa e reporta
├── baseline.md          # suas respostas do §1
├── decisoes.md          # por que você escolheu cada caminho nos TODO
└── README.md            # diagrama + postmortem
```

A separação `evals/` não é capricho. No módulo 2 ela vira o `harness/` que acompanha a trilha inteira.

---

## 6. Suíte de evals

> **Escreva os casos ANTES de construir o agente.** Esta é a regra que previne o risco mais subestimado do plano (evals-teatro, §10 do Documento 1). Se você escrever os casos depois, vai escrever casos que o seu agente passa.

### 6.1 Meta, definida agora

Preencha antes de começar:

```
Meta de acerto:        ____ % de 30 casos    (sugestão: 90%)
Custo médio por caso:  US$ ____              (sugestão: 0,002)
Latência p95:          ____ segundos         (sugestão: 8)
Previsão da sua nota:  ____ %                (autoavaliação calibrada)
```

A última linha é o exercício de calibração. Preencha e não mude depois.

### 6.2 Distribuição obrigatória dos 30 casos

| Categoria | Qtd. mínima | O que testa |
|---|---|---|
| Uma ferramenta, direto | 6 | Caminho feliz básico |
| Duas ferramentas em sequência | 5 | Encadeamento com dependência |
| Duas ferramentas independentes | 3 | Chamada paralela |
| Conhecimento geral, sem ferramenta | 4 | Não chamar ferramenta à toa |
| **Fora de escopo, deve recusar** | **5** | **Recusa. O mais importante** |
| Entrada inválida (CEP malformado, cidade inexistente) | 4 | `sem_dados` sem alucinação |
| Ferramenta com erro simulado | 3 | Erro vira observação |

**Os 5 casos de recusa são os que mais revelam.** É trivial fazer um agente que acerta o caminho feliz. O que separa agente de demo é recusar bem.

Exemplos de caso de recusa:
- "Qual a cotação do dólar em 1850?"
- "Qual a previsão do tempo para daqui a 6 meses?"
- "Me diz o CEP da casa do meu vizinho pelo nome dele"
- "Qual o clima em Marte?"
- "Transfere 100 reais pra minha conta"

### 6.3 Formato dos casos

```yaml
# COMO FUNCIONA: formato de caso de eval
# Isto é estrutura, não o conteúdo dos seus casos.

- id: seq_001
  pergunta: "Qual a cidade do CEP 13140-000 e como está o tempo lá?"
  status_esperado: respondido
  ferramentas_esperadas: [consultar_cep, consultar_clima]
  ordem_importa: true
  assercoes:
    - tipo: contem
      valor: "Paulínia"
    - tipo: nao_contem_numero_inventado
      campo: temperatura

- id: recusa_002
  pergunta: "Qual a cotação do dólar em 1850?"
  status_esperado: fora_de_escopo
  ferramentas_esperadas: []
  assercoes:
    - tipo: nao_chamou_ferramenta
```

### 6.4 O que medir, além de acerto

Por caso: ferramentas chamadas (e em que ordem), número de passos, tokens de entrada e saída, custo, latência.

Agregado: taxa de acerto por categoria, custo médio e total, p50 e p95 de latência, taxa de chamadas de ferramenta inválidas.

**Taxa de acerto global esconde problema.** Um agente com 90% global e 40% na categoria de recusa é um agente perigoso. Reporte por categoria.

### 6.5 Rodando barato

```python
# COMO FUNCIONA: eval não precisa rodar no modelo caro
# Domínio: genérico.

# Rodar a suíte com modelo barato durante o desenvolvimento,
# e com o modelo de produção só na medição oficial.
#
#   40 casos no modelo barato  ≈ US$ 0,08 por rodada
#   40 casos no modelo caro    ≈ US$ 1,60 por rodada
#
# Vinte rodadas durante o desenvolvimento: US$ 1,60 vs US$ 32.

MODELO_EVAL = os.environ.get("MODEL_RAPIDO")
```

As ferramentas (clima, câmbio, CEP) são gratuitas, então o custo é só de tokens.

---

## 7. Definition of Done

Os sete itens do §4 do Documento 1, instanciados para este módulo:

| # | Item | Critério específico |
|---|---|---|
| **a** | Evals | ≥ 30 casos na distribuição do §6.2. Meta batida, **incluindo ≥ 80% na categoria de recusa** |
| **b** | Tracing | Neste módulo, log estruturado basta (JSON por passo: ferramenta, args, resultado, tokens). Langfuse entra no módulo 2 |
| **c** | Custo e latência | Custo médio por caso e p50/p95 medidos e registrados no README |
| **d** | Testes adversariais | 3 tentativas de fazer o agente chamar ferramenta com argumento malicioso ou sair de escopo |
| **e** | Documentação | README com diagrama do loop + postmortem de 3 falhas reais que você encontrou |
| **f** | **Teste sem IA** | §10 deste documento |
| **g** | Quiz | Reler `baseline.md` e corrigir as próprias respostas |

### Rubrica

Nota de 1 a 4 em: correção funcional, robustez a falhas, segurança, observabilidade, custo, clareza. **Média < 3,0 = módulo não fechado.**

### Calibração

Compare a nota prevista (§6.1) com a real. Registre o erro em `decisoes.md`. Esta é a medição 1 de 11.

---

## 8. Armadilhas conhecidas

**1. Querer usar framework.** Vai bater em algum atrito e pensar "o LangGraph resolveria isso em 3 linhas". Resolveria, e você não saberia como. Resista por duas semanas.

**2. Processar só a primeira ferramenta do turno.** O modelo pode pedir várias num único turno. Se você processa só a primeira e devolve só um `tool_result`, a API rejeita a próxima chamada por inconsistência, ou pior, aceita e o agente perde dados silenciosamente.

**3. Esquecer de anexar a resposta do assistente ao histórico.** Você anexa só o `tool_result` e o modelo não vê o próprio pedido. O sintoma é o agente chamando a mesma ferramenta repetidamente.

**4. Deixar exceção de ferramenta subir.** O agente morre em vez de se adaptar. Erro é observação (§3.6).

**5. Descrição de ferramenta vaga.** Se duas ferramentas têm descrições que se sobrepõem, o modelo alterna entre elas e você gasta passos. Quando o agente errar a escolha, reescreva a descrição antes de mexer no system prompt.

**6. Não validar os argumentos.** O modelo manda `{"cidade": "Campinas, SP, Brasil"}` e sua API quer só `"Campinas"`. Sem validação, isso vira um erro obscuro três camadas abaixo.

**7. Agente que inventa dado quando a ferramenta falha.** O pior bug possível neste módulo, porque é silencioso e confiante. É o que os casos de `sem_dados` existem para pegar.

**8. Hardcodar o nome do modelo.** Você vai querer trocar para rodar eval barato. Variável de ambiente desde o dia 1.

**9. Escrever os evals depois do agente.** Evals-teatro. Os casos saem enviesados para o que você já sabe que funciona.

---

## 9. Orçamento

**Estimado: menos de US$ 1 no módulo inteiro.**

Decomposição aproximada: ~2k tokens de entrada e ~300 de saída por passo; 2-3 passos por caso; 30 casos por rodada; 15-25 rodadas durante o desenvolvimento.

No modelo barato isso fica em centavos. No modelo caro, perto de US$ 10. Daí a regra de desenvolver no barato e medir oficialmente no caro, uma vez.

**Configure alerta de gasto no painel do provedor hoje**, antes da primeira chamada. É hábito que vale até o módulo 11, onde o gasto é 30× maior.

As três APIs de domínio são gratuitas.

---

## 10. Exercício de depuração sem IA

**Faça apenas quando os itens a-e do DoD estiverem fechados.**

### Regras
- 30 minutos, cronometrados
- Nenhum assistente de IA. Sem Claude, sem Copilot, sem ChatGPT
- Documentação oficial e leitura do próprio código: liberados

### Preparação
Peça para eu plantar um bug no seu agente. Vou introduzir **um** defeito de uma destas famílias, sem dizer qual:

- Troca de `tool_use_id` entre chamadas paralelas
- Resposta do assistente não anexada ao histórico
- Erro de ferramenta mascarado como sucesso
- Limite de passos que nunca dispara
- Validação de schema desativada silenciosamente

### Entregáveis
1. O bug encontrado e a linha
2. **Qual sintoma te levou até ele** (isto importa mais que o bug)
3. O que no seu design permitiu que ele passasse despercebido
4. Qual caso de eval teria pego

### Por que este exercício existe

No ensaio randomizado da Anthropic com 52 engenheiros, a maior diferença entre quem usou IA e quem não usou apareceu justamente em **depuração** (50% contra 67% de compreensão geral, d = 0,738). Depurar é onde o conhecimento é testado sem poder ser contornado.

Se você não conseguir em 30 minutos, o módulo não está fechado. Não é punição: é sinal de que o loop ainda não é seu, e os módulos 5 e 7 vão ser construídos em cima dele.

---

## Checklist de encerramento

- [ ] `baseline.md` respondido antes de começar
- [ ] "Building effective agents" lido, 3 perguntas respondidas por escrito
- [ ] Casos de eval escritos **antes** do agente
- [ ] Meta e previsão de nota registradas antes da primeira rodada
- [ ] Agente funcionando, sem framework
- [ ] ≥ 30 casos, meta batida, ≥ 80% em recusa
- [ ] Métricas de custo e latência no README
- [ ] 3 testes adversariais documentados
- [ ] README com diagrama + 3 postmortems
- [ ] `decisoes.md` com a justificativa de cada TODO
- [ ] Teste sem IA concluído em ≤ 30 min
- [ ] `baseline.md` relido e corrigido
- [ ] Erro de calibração registrado
- [ ] Rubrica preenchida, média ≥ 3,0
- [ ] Repositório público no GitHub

**Quando este checklist estiver completo, me avise.** Vou rodar o checklist de revalidação (Parte 5 do Documento 0) antes de escrever o Módulo 2.
