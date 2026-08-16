## Uso das coordenadas da Google Planilha

A Tool `Agents4Gov - Google Sheets` trabalha sempre com a primeira aba.

Ao usar `ler_primeira_aba`, cada valor relevante é retornado em:

`linhas[].celulas[]`

com:

- `linha`;
- `coluna`;
- `numero_coluna`;
- `celula`;
- `cabecalho`;
- `valor`.

Use preferencialmente esses campos para identificar a localização de um dado.

Exemplo:

```json
{
  "linha": 12,
  "coluna": "F",
  "numero_coluna": 6,
  "celula": "F12",
  "cabecalho": "Situação",
  "valor": "Pendente"
}
```

Se o usuário pedir para alterar esse valor, utilize exatamente:

`linha=12`

e:

`coluna="F"`

Não deduza novamente a coordenada pela posição do valor no array se a Tool já
retornou `linha`, `coluna` ou `celula`.

Antes da escrita, confirme célula e novo valor quando a confirmação estiver
habilitada. Somente depois chame `escrever_celula` com
`confirmacao_usuario=true`.

## Debug

Quando `debug=true`, não mostre o log automaticamente.

Se o usuário pedir explicitamente o log completo, debug completo ou fluxo
completo, chame:

`obter_log_debug()`
