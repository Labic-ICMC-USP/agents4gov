# Agents4Gov - Integrando com Browser via Playwright MCP

Este tutorial mostra como integrar um **Open WebUI já existente em Docker** com um navegador controlado por agentes usando o **Playwright MCP**.

A arquitetura final será:

```text
Open WebUI
    |
    | MCP Streamable HTTP
    v
MCP_HOST:MCP_PORT
    |
    v
Playwright MCP (Docker)
    |
    v
Chromium isolado e efêmero
```

O objetivo é:

- disponibilizar ferramentas de navegador para agentes no Open WebUI;
- não persistir cookies, sessões ou histórico entre execuções;
- permitir múltiplos usuários sem compartilhar o estado do navegador;
- expor o Playwright MCP em uma porta TCP própria;
- controlar o acesso à porta usando `iptables`;
- iniciar automaticamente as regras de firewall com `systemd`;
- não usar Nginx;
- não usar autenticação Bearer no Playwright MCP;
- configurar o Open WebUI com `Auth=None`.

---

## 1. Variáveis usadas neste tutorial

Antes de começar, escolha os valores do seu ambiente.

Exemplo conceitual:

```bash
MCP_HOST="mcp.exemplo.interno"
MCP_PORT="35010"
MCP_PUBLIC_IP="AAA.BBB.CCC.DDD"
```

As redes internas autorizadas também devem ser adaptadas ao seu ambiente:

```bash
LOCAL_NET="127.0.0.0/8"
DOCKER_NET_1="172.17.0.0/16"
DOCKER_NET_2="172.18.0.0/16"
```

Neste tutorial, considere:

```text
MCP_HOST       = domínio ou hostname do servidor Playwright MCP
MCP_PORT       = porta TCP pública do MCP
MCP_PUBLIC_IP  = IP do próprio servidor MCP
DOCKER_NET_1   = primeira rede interna autorizada
DOCKER_NET_2   = segunda rede interna autorizada
```

---

# 2. Verificar o Docker

O Open WebUI já deve estar rodando em Docker.

No servidor onde será instalado o Playwright MCP:

```bash
docker --version
```

Confira os containers existentes:

```bash
docker ps
```

---

# 3. Baixar o Playwright MCP

Baixe a imagem oficial:

```bash
sudo docker pull mcr.microsoft.com/playwright/mcp
```

Caso já exista um container anterior com o mesmo nome:

```bash
sudo docker rm -f playwright-mcp 2>/dev/null || true
```

---

# 4. Executar o Playwright MCP

Defina a porta desejada no shell:

```bash
MCP_PORT="35010"
```

Inicie o container:

```bash
sudo docker run -d \
  --name playwright-mcp \
  --restart unless-stopped \
  --init \
  --network host \
  --entrypoint node \
  mcr.microsoft.com/playwright/mcp \
  /app/cli.js \
  --headless \
  --browser chromium \
  --no-sandbox \
  --host 0.0.0.0 \
  --port "$MCP_PORT" \
  --isolated \
  --image-responses allow \
  --allowed-hosts '*'
```

## Por que usamos `--isolated`

O parâmetro:

```text
--isolated
```

faz com que o estado do navegador seja temporário.

Não estamos usando:

```text
--user-data-dir
--storage-state
--shared-browser-context
```

Com isso, cookies, autenticação, `localStorage` e demais dados de navegação não devem ser reutilizados entre sessões independentes.

Também usamos:

```text
--allowed-hosts '*'
```

porque o controle de acesso será feito no firewall.

`--allowed-hosts` não é autenticação de clientes MCP.

---

# 5. Verificar o Playwright MCP

Confira o container:

```bash
sudo docker ps | grep playwright-mcp
```

Veja os logs:

```bash
sudo docker logs --tail 100 playwright-mcp
```

Confira se a porta está aberta:

```bash
sudo ss -ltnp | grep ":${MCP_PORT}"
```

---

# 6. Teste HTTP local

Teste inicialmente no próprio servidor MCP:

```bash
curl -i --max-time 5 "http://127.0.0.1:${MCP_PORT}/mcp"
```

Um `GET` simples pode retornar `400 Bad Request`.

Isso não significa que o MCP esteja com problema. O endpoint `/mcp` espera mensagens do protocolo MCP.

---

# 7. Teste de handshake MCP

Defina a URL:

```bash
MCP_HOST="mcp.exemplo.interno"
```

```bash
MCP_URL="http://${MCP_HOST}:${MCP_PORT}/mcp"
```

Faça a inicialização MCP:

```bash
curl -sS \
  -D /tmp/playwright-mcp-headers.txt \
  -o /tmp/playwright-mcp-init.txt \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-11-25",
      "capabilities": {},
      "clientInfo": {
        "name": "curl-test",
        "version": "1.0"
      }
    }
  }'
```

Veja os headers:

```bash
cat /tmp/playwright-mcp-headers.txt
```

Veja a resposta:

```bash
cat /tmp/playwright-mcp-init.txt
```

---

# 8. Obter o identificador da sessão MCP

Extraia o `Mcp-Session-Id`:

```bash
SESSION_ID=$(awk 'BEGIN{IGNORECASE=1} /^mcp-session-id:/ {gsub("\r","",$2); print $2}' /tmp/playwright-mcp-headers.txt)
```

Confira:

```bash
echo "$SESSION_ID"
```

---

# 9. Finalizar a inicialização MCP

Envie a notificação `initialized`:

```bash
curl -i -sS \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  --data '{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
  }'
```

---

# 10. Listar as ferramentas disponíveis

Liste as ferramentas MCP:

```bash
curl -sS \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  --data '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }'
```

Para visualizar apenas os nomes das ferramentas de navegador:

```bash
curl -sS \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  --data '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }' | grep -o '"name":"browser_[^"]*"'
```

Devem aparecer ferramentas como:

```text
browser_navigate
browser_snapshot
browser_click
browser_fill_form
browser_take_screenshot
browser_evaluate
browser_file_upload
browser_close
```

---

# 11. Testar uma navegação real

Abra uma página:

```bash
curl -sS \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  --data '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "browser_navigate",
      "arguments": {
        "url": "https://example.com"
      }
    }
  }'
```

---

# 12. Testar JavaScript no navegador

Leia o título da página:

```bash
curl -sS \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  --data '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "browser_evaluate",
      "arguments": {
        "function": "() => document.title"
      }
    }
  }'
```

O resultado esperado para `example.com` contém:

```text
Example Domain
```

---

# 13. Fechar o navegador

Ao terminar a tarefa:

```bash
curl -sS \
  -X POST "$MCP_URL" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  --data '{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {
      "name": "browser_close",
      "arguments": {}
    }
  }'
```

Opcionalmente encerre também a sessão MCP:

```bash
curl -i -X DELETE "$MCP_URL" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'MCP-Protocol-Version: 2025-11-25'
```

---

# 14. Firewall dedicado para o Playwright MCP

Vamos criar regras que afetam exclusivamente:

```text
TCP/MCP_PORT
```

Todas as regras `ACCEPT` e `DROP` terão explicitamente:

```text
-p tcp --dport MCP_PORT
```

Crie o script:

```bash
sudo nano /usr/local/sbin/playwright-mcp-firewall.sh
```

Cole:

```bash
#!/usr/bin/env bash

set -euo pipefail

PORT="35010"
CHAIN="PLAYWRIGHT_MCP"
IPTABLES="/usr/sbin/iptables"

LOCAL_NET="127.0.0.0/8"
DOCKER_NET_1="172.17.0.0/16"
DOCKER_NET_2="172.18.0.0/16"
MCP_PUBLIC_IP="AAA.BBB.CCC.DDD/32"

start_firewall() {
    echo "[INFO] Configurando firewall exclusivamente para TCP/${PORT}..."

    if ! "$IPTABLES" -w -nL "$CHAIN" >/dev/null 2>&1; then
        "$IPTABLES" -w -N "$CHAIN"
    fi

    "$IPTABLES" -w -F "$CHAIN"

    "$IPTABLES" -w -A "$CHAIN" \
        -p tcp \
        -s "$LOCAL_NET" \
        --dport "$PORT" \
        -j ACCEPT

    "$IPTABLES" -w -A "$CHAIN" \
        -p tcp \
        -s "$DOCKER_NET_1" \
        --dport "$PORT" \
        -j ACCEPT

    "$IPTABLES" -w -A "$CHAIN" \
        -p tcp \
        -s "$DOCKER_NET_2" \
        --dport "$PORT" \
        -j ACCEPT

    "$IPTABLES" -w -A "$CHAIN" \
        -p tcp \
        -s "$MCP_PUBLIC_IP" \
        --dport "$PORT" \
        -j ACCEPT

    "$IPTABLES" -w -A "$CHAIN" \
        -p tcp \
        --dport "$PORT" \
        -j DROP

    while "$IPTABLES" -w -C INPUT \
        -p tcp \
        --dport "$PORT" \
        -j "$CHAIN" 2>/dev/null
    do
        "$IPTABLES" -w -D INPUT \
            -p tcp \
            --dport "$PORT" \
            -j "$CHAIN"
    done

    "$IPTABLES" -w -I INPUT 1 \
        -p tcp \
        --dport "$PORT" \
        -j "$CHAIN"

    echo "[OK] Firewall configurado exclusivamente para TCP/${PORT}."
}

stop_firewall() {
    echo "[INFO] Removendo regras exclusivamente de TCP/${PORT}..."

    while "$IPTABLES" -w -C INPUT \
        -p tcp \
        --dport "$PORT" \
        -j "$CHAIN" 2>/dev/null
    do
        "$IPTABLES" -w -D INPUT \
            -p tcp \
            --dport "$PORT" \
            -j "$CHAIN"
    done

    if "$IPTABLES" -w -nL "$CHAIN" >/dev/null 2>&1; then
        "$IPTABLES" -w -F "$CHAIN"
        "$IPTABLES" -w -X "$CHAIN"
    fi

    echo "[OK] Regras de TCP/${PORT} removidas."
}

status_firewall() {
    echo
    echo "=== INPUT / TCP ${PORT} ==="

    "$IPTABLES" -w -L INPUT -n -v --line-numbers \
        | grep -E "Chain|${CHAIN}|dpt:${PORT}" || true

    echo
    echo "=== ${CHAIN} ==="

    "$IPTABLES" -w -L "$CHAIN" -n -v --line-numbers 2>/dev/null \
        || echo "Chain ${CHAIN} não existe."
}

case "${1:-start}" in
    start)
        start_firewall
        ;;
    stop)
        stop_firewall
        ;;
    restart)
        stop_firewall
        start_firewall
        ;;
    status)
        status_firewall
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
```

Edite no script:

```text
PORT
MCP_PUBLIC_IP
DOCKER_NET_1
DOCKER_NET_2
```

de acordo com seu ambiente.

---

# 15. Tornar o script executável

```bash
sudo chmod 755 /usr/local/sbin/playwright-mcp-firewall.sh
```

Teste:

```bash
sudo /usr/local/sbin/playwright-mcp-firewall.sh restart
```

Veja o estado:

```bash
sudo /usr/local/sbin/playwright-mcp-firewall.sh status
```

Também pode conferir diretamente:

```bash
sudo iptables -L PLAYWRIGHT_MCP -n -v --line-numbers
```

Todas as regras `ACCEPT` e `DROP` devem mostrar:

```text
tcp dpt:35010
```

ou a porta que você configurou.

---

# 16. Criar serviço systemd para o firewall

Crie:

```bash
sudo nano /etc/systemd/system/playwright-mcp-firewall.service
```

Cole:

```ini
[Unit]
Description=Firewall exclusivo para Playwright MCP
Wants=network-online.target
After=network-online.target
Before=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/playwright-mcp-firewall.sh start
ExecReload=/usr/local/sbin/playwright-mcp-firewall.sh restart
ExecStop=/usr/local/sbin/playwright-mcp-firewall.sh stop
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Recarregue o systemd:

```bash
sudo systemctl daemon-reload
```

Habilite no boot:

```bash
sudo systemctl enable playwright-mcp-firewall.service
```

Inicie:

```bash
sudo systemctl restart playwright-mcp-firewall.service
```

Confira:

```bash
sudo systemctl status playwright-mcp-firewall.service
```

Verifique se está habilitado:

```bash
sudo systemctl is-enabled playwright-mcp-firewall.service
```

---

# 17. Testar o firewall

No próprio servidor:

```bash
curl -i --max-time 5 "http://127.0.0.1:${MCP_PORT}/mcp"
```

Teste usando o endereço configurado:

```bash
curl -i --max-time 5 "http://${MCP_HOST}:${MCP_PORT}/mcp"
```

Uma origem não autorizada deverá sofrer `DROP`, normalmente resultando em timeout.

Veja os contadores:

```bash
sudo iptables -L PLAYWRIGHT_MCP -n -v --line-numbers
```

---

# 18. Testar a partir do servidor Open WebUI

Antes de cadastrar o MCP no Open WebUI, teste a partir da máquina onde roda o backend do Open WebUI:

```bash
curl -i --connect-timeout 5 "http://${MCP_HOST}:${MCP_PORT}/mcp"
```

Receber `400 Bad Request` em um `GET` simples é suficiente para demonstrar que:

```text
Open WebUI host
    |
    v
firewall
    |
    v
Playwright MCP
```

está acessível.

Se der timeout, a origem usada pelo servidor Open WebUI não está autorizada pelo firewall.

---

# 19. Configurar o MCP no Open WebUI

No Open WebUI, entre como administrador.

Vá para:

```text
Admin Settings
→ Integrations
→ Add Server
```

Configure:

```text
Name:
Playwright Browser
```

```text
Type:
MCP (Streamable HTTP)
```

```text
URL:
http://MCP_HOST:MCP_PORT/mcp
```

Exemplo conceitual:

```text
http://mcp.exemplo.interno:35010/mcp
```

O ponto mais importante da configuração é:

```text
Auth:
None
```

Não configure:

```text
Bearer
```

com chave vazia.

Esse cenário pode resultar em:

```text
Verify Connection: OK
```

mas, ao usar a ferramenta no chat:

```text
Failed to connect to MCP server
```

O Playwright MCP desta configuração não está usando API Key nem Bearer token.

---

# 20. Verificar a integração

Use:

```text
Verify Connection
```

O Open WebUI deve conseguir consultar o servidor MCP e descobrir suas ferramentas.

---

# 21. Disponibilizar a integração aos usuários

Configure o **Access Control** da integração de acordo com os usuários ou grupos desejados.

Depois, em um chat novo, habilite:

```text
Integrations
→ Tools
→ Playwright Browser
```

---

# 22. Primeiro teste no chat

Envie:

```text
Use obrigatoriamente o Playwright Browser.
Abra https://example.com, informe o título da página
e depois feche o navegador.
```

O agente deverá utilizar ferramentas como:

```text
browser_navigate
browser_snapshot
browser_evaluate
browser_close
```

---

# 23. Testar screenshot

Envie:

```text
Abra https://example.com usando o Playwright Browser.
Tire um screenshot da página e depois feche o navegador.
```

O agente poderá utilizar:

```text
browser_take_screenshot
```

---

# 24. Regra recomendada para o System Prompt

Para modelos que tenham acesso ao Playwright Browser, recomenda-se incluir:

```text
Ao utilizar o Playwright Browser, considere o navegador temporário
e exclusivo da tarefa atual.

Não reutilize autenticação, cookies, localStorage ou informações
de sessões anteriores.

Ao concluir uma tarefa de navegação, execute browser_close,
inclusive quando ocorrer erro durante a execução.
```

---

# 25. Testar isolamento entre usuários

Antes de usar sistemas autenticados em produção, faça um teste com dois usuários simultâneos.

## Usuário A

Peça:

```text
Abra https://example.com.

Use browser_evaluate para definir:

localStorage.setItem("teste_usuario", "USUARIO_A")

Não feche o browser ainda.
```

## Usuário B

Em outra conta do Open WebUI:

```text
Abra https://example.com.

Use browser_evaluate para retornar:

localStorage.getItem("teste_usuario")
```

O resultado esperado para o usuário B é:

```text
null
```

Depois feche os dois browsers.

---

# 26. Logs para diagnóstico

Logs do Playwright MCP:

```bash
sudo docker logs --since 10m -f playwright-mcp
```

Se o Open WebUI também estiver em Docker, descubra o nome:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}'
```

Depois acompanhe seus logs:

```bash
sudo docker logs --since 10m -f NOME_DO_CONTAINER_OPENWEBUI
```

Filtrando mensagens relevantes:

```bash
sudo docker logs --since 10m NOME_DO_CONTAINER_OPENWEBUI 2>&1 | grep -iE 'mcp|tool|playwright|error|exception|failed'
```

Também confira os contadores do firewall:

```bash
sudo iptables -L PLAYWRIGHT_MCP -n -v --line-numbers
```

---

# 27. Comandos úteis do Playwright MCP

Status:

```bash
sudo docker ps | grep playwright-mcp
```

Logs:

```bash
sudo docker logs -f playwright-mcp
```

Reiniciar:

```bash
sudo docker restart playwright-mcp
```

Parar:

```bash
sudo docker stop playwright-mcp
```

Iniciar:

```bash
sudo docker start playwright-mcp
```

---

# 28. Comandos úteis do firewall

Status:

```bash
sudo /usr/local/sbin/playwright-mcp-firewall.sh status
```

Reaplicar:

```bash
sudo systemctl restart playwright-mcp-firewall.service
```

Ver regras:

```bash
sudo iptables -L PLAYWRIGHT_MCP -n -v --line-numbers
```

Ver referência na INPUT:

```bash
sudo iptables -L INPUT -n -v --line-numbers
```

---

# 29. Rollback do firewall

Para remover apenas as regras gerenciadas pelo script:

```bash
sudo /usr/local/sbin/playwright-mcp-firewall.sh stop
```

Para impedir que voltem no próximo boot:

```bash
sudo systemctl disable playwright-mcp-firewall.service
```

---

# 30. Arquitetura final

```text
                    Open WebUI
                        |
                        | MCP Streamable HTTP
                        |
                        v
              MCP_HOST:MCP_PORT
                        |
                iptables TCP/MCP_PORT
                        |
         +--------------+--------------+
         |              |              |
      ACCEPT          ACCEPT          DROP
     localhost       redes internas   demais
                        |
                        v
                Playwright MCP
                        |
                 --isolated
                        |
                        v
                    Chromium
                        |
               browser temporário
```

O Open WebUI passa a oferecer aos agentes ferramentas de navegação genéricas, enquanto fluxos específicos podem posteriormente ser implementados como **Workspace Tools** que executam automações determinísticas sobre o browser.
