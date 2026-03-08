#!/bin/bash
# VIRON Debug Script — Run after changes to check everything
# Usage: bash scripts/debug_viron.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "╔══════════════════════════════════════════╗"
echo "║        🤖 VIRON Debug Report             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 1. Check Python syntax
echo -e "${YELLOW}[1/8] Checking Python syntax...${NC}"
cd ~/VIRON
python3 -c "import ast; ast.parse(open('voice_pipeline.py').read())" 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✅ voice_pipeline.py — OK${NC}"
else
    echo -e "  ${RED}❌ voice_pipeline.py — SYNTAX ERROR${NC}"
fi

python3 -c "import ast; ast.parse(open('backend/server.py').read())" 2>&1
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✅ backend/server.py — OK${NC}"
else
    echo -e "  ${RED}❌ backend/server.py — SYNTAX ERROR${NC}"
fi

# 2. Check HTML braces balance
echo -e "${YELLOW}[2/8] Checking HTML/JS syntax...${NC}"
python3 -c "
with open('viron-complete.html') as f:
    c = f.read()
s = c[c.find('<script>'):c.rfind('</script>')]
o = s.count('{')
c2 = s.count('}')
if o == c2:
    print(f'  ✅ viron-complete.html — Braces balanced ({o}/{c2})')
else:
    print(f'  ❌ viron-complete.html — Braces MISMATCH ({o} open, {c2} close)')
"

# 3. Check services
echo -e "${YELLOW}[3/8] Checking services...${NC}"
for svc in viron-backend viron-pipeline; do
    status=$(systemctl is-active $svc 2>/dev/null)
    if [ "$status" = "active" ]; then
        echo -e "  ${GREEN}✅ $svc — running${NC}"
    else
        echo -e "  ${RED}❌ $svc — $status${NC}"
    fi
done

# 4. Check ports
echo -e "${YELLOW}[4/8] Checking ports...${NC}"
for port in 5000 8085; do
    if ss -tln | grep -q ":$port "; then
        echo -e "  ${GREEN}✅ Port $port — listening${NC}"
    else
        echo -e "  ${RED}❌ Port $port — NOT listening${NC}"
    fi
done

# 5. Check kiosk
echo -e "${YELLOW}[5/8] Checking kiosk...${NC}"
if pgrep -f viron_kiosk > /dev/null; then
    echo -e "  ${GREEN}✅ viron_kiosk — running${NC}"
else
    echo -e "  ${RED}❌ viron_kiosk — NOT running${NC}"
fi

# 6. Check API endpoints
echo -e "${YELLOW}[6/8] Checking API endpoints...${NC}"
resp=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/ 2>/dev/null)
if [ "$resp" = "200" ]; then
    echo -e "  ${GREEN}✅ Backend (5000) — HTTP $resp${NC}"
else
    echo -e "  ${RED}❌ Backend (5000) — HTTP $resp${NC}"
fi

resp=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8085/pipeline/state 2>/dev/null)
if [ "$resp" = "200" ]; then
    state=$(curl -s http://localhost:8085/pipeline/state 2>/dev/null)
    echo -e "  ${GREEN}✅ Pipeline (8085) — $state${NC}"
else
    echo -e "  ${RED}❌ Pipeline (8085) — HTTP $resp${NC}"
fi

# 7. Check mic
echo -e "${YELLOW}[7/8] Checking audio...${NC}"
if arecord -l 2>/dev/null | grep -q "card"; then
    echo -e "  ${GREEN}✅ Mic — available${NC}"
    arecord -l 2>/dev/null | grep "card" | head -2 | while read line; do
        echo "     $line"
    done
else
    echo -e "  ${RED}❌ Mic — NOT found${NC}"
fi

if aplay -l 2>/dev/null | grep -q "card"; then
    echo -e "  ${GREEN}✅ Speaker — available${NC}"
else
    echo -e "  ${RED}❌ Speaker — NOT found${NC}"
fi

# 8. Check .env
echo -e "${YELLOW}[8/8] Checking .env keys...${NC}"
for key in OPENAI_API_KEY DEEPGRAM_API_KEY GROQ_API_KEY ANTHROPIC_API_KEY PICOVOICE_ACCESS_KEY; do
    val=$(grep "^$key=" ~/VIRON/.env 2>/dev/null | cut -d= -f2)
    if [ -n "$val" ] && [ "$val" != "" ]; then
        echo -e "  ${GREEN}✅ $key — set (${#val} chars)${NC}"
    else
        echo -e "  ${RED}❌ $key — MISSING${NC}"
    fi
done

# 9. Recent errors
echo ""
echo -e "${YELLOW}Recent pipeline errors (last 20 lines):${NC}"
sudo journalctl -u viron-pipeline --no-pager -n 20 --output=short-iso 2>/dev/null | grep -i "error\|warning\|fail\|crash\|stuck\|exception" | tail -10
if [ $? -ne 0 ]; then
    echo -e "  ${GREEN}No recent errors found${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        Debug complete!                   ║"
echo "╚══════════════════════════════════════════╝"
