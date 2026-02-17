#!/bin/bash
# ============================================
# VIRON Boot Splash Setup
# Hides Ubuntu boot screen, shows VIRON logo
# ============================================

echo "🎨 Setting up VIRON Boot Splash..."

# 1. Create Plymouth theme directory
THEME_DIR="/usr/share/plymouth/themes/viron"
sudo mkdir -p "$THEME_DIR"

# 2. Create the theme descriptor
sudo tee "$THEME_DIR/viron.plymouth" > /dev/null <<'EOF'
[Plymouth Theme]
Name=VIRON AI Tutor
Description=Boot splash for VIRON AI Tutor Robot
ModuleName=script

[script]
ImageDir=/usr/share/plymouth/themes/viron
ScriptFile=/usr/share/plymouth/themes/viron/viron.script
EOF

# 3. Create the animation script
sudo tee "$THEME_DIR/viron.script" > /dev/null <<'SCRIPTEOF'
// VIRON Boot Splash Animation
Window.SetBackgroundTopColor(0, 0, 0);
Window.SetBackgroundBottomColor(0, 0, 0);

// Load logo
logo.image = Image("viron-logo.png");
logo.sprite = Sprite(logo.image);
logo.sprite.SetX(Window.GetWidth() / 2 - logo.image.GetWidth() / 2);
logo.sprite.SetY(Window.GetHeight() / 2 - logo.image.GetHeight() / 2 - 40);

// Loading text
loading_text.image = Image.Text("VIRON AI Tutor • Initializing...", 0.6, 0.8, 1, 1, "Sans 14");
loading_text.sprite = Sprite(loading_text.image);
loading_text.sprite.SetX(Window.GetWidth() / 2 - loading_text.image.GetWidth() / 2);
loading_text.sprite.SetY(Window.GetHeight() / 2 + 80);

// Animated dots
progress = 0;
fun refresh_callback() {
    progress++;
    dots = "";
    for (i = 0; i < (progress / 15 % 4); i++) dots += " •";
    
    t.image = Image.Text("Loading" + dots, 0.4, 0.6, 0.8, 1, "Sans 12");
    t.sprite = Sprite(t.image);
    t.sprite.SetX(Window.GetWidth() / 2 - t.image.GetWidth() / 2);
    t.sprite.SetY(Window.GetHeight() / 2 + 110);

    // Pulsing glow effect on logo
    opacity = 0.7 + Math.Sin(progress * 0.05) * 0.3;
    logo.sprite.SetOpacity(opacity);
}
Plymouth.SetRefreshFunction(refresh_callback);

// Hide all password/message prompts during boot
fun message_callback(text) { /* suppress */ }
Plymouth.SetMessageFunction(message_callback);

fun display_password_callback(prompt, bullets) {
    p.image = Image.Text(prompt, 1, 1, 1, 1, "Sans 14");
    p.sprite = Sprite(p.image);
    p.sprite.SetX(Window.GetWidth() / 2 - p.image.GetWidth() / 2);
    p.sprite.SetY(Window.GetHeight() / 2 + 50);
}
Plymouth.SetDisplayPasswordFunction(display_password_callback);
SCRIPTEOF

# 4. Generate VIRON logo (SVG to PNG)
# Create a simple logo using ImageMagick
if command -v convert &> /dev/null; then
    convert -size 200x200 xc:transparent \
        -fill "rgba(0,200,255,0.15)" -draw "circle 100,100 100,20" \
        -fill "rgba(0,200,255,0.3)" -draw "circle 100,100 100,40" \
        -font "DejaVu-Sans-Bold" -pointsize 48 -fill "rgb(0,200,255)" \
        -gravity center -annotate +0-10 "V" \
        -font "DejaVu-Sans" -pointsize 16 -fill "rgb(180,220,255)" \
        -gravity center -annotate +0+35 "VIRON" \
        "$THEME_DIR/viron-logo.png"
    echo "✅ Logo generated"
else
    # Fallback: create minimal PNG with Python
    python3 -c "
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGBA', (200, 200), (0,0,0,0))
d = ImageDraw.Draw(img)
d.ellipse([20,20,180,180], fill=(0,50,80,60), outline=(0,200,255,100), width=3)
d.ellipse([40,40,160,160], fill=(0,30,50,40))
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 60)
    sfont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
except: font = ImageFont.load_default(); sfont = font
d.text((100,70), 'V', fill=(0,200,255), font=font, anchor='mm')
d.text((100,120), 'VIRON', fill=(180,220,255), font=sfont, anchor='mm')
img.save('$THEME_DIR/viron-logo.png')
print('Logo created')
" 2>/dev/null || echo "⚠ Could not generate logo. Place viron-logo.png manually in $THEME_DIR"
fi

# 5. Install the theme
sudo update-alternatives --install /usr/share/plymouth/themes/default.plymouth default.plymouth "$THEME_DIR/viron.plymouth" 200
sudo update-alternatives --set default.plymouth "$THEME_DIR/viron.plymouth"

# 6. Configure GRUB to hide text
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash loglevel=0 vt.global_cursor_default=0"/' /etc/default/grub
sudo sed -i 's/#GRUB_HIDDEN_TIMEOUT=.*/GRUB_HIDDEN_TIMEOUT=0/' /etc/default/grub 2>/dev/null
# Hide GRUB menu completely
if ! grep -q "GRUB_TIMEOUT_STYLE" /etc/default/grub; then
    echo 'GRUB_TIMEOUT_STYLE=hidden' | sudo tee -a /etc/default/grub > /dev/null
fi
sudo update-grub 2>/dev/null

# 7. Update initramfs to include Plymouth
sudo update-initramfs -u

# 8. Disable login screen (auto-login)
LIGHTDM_CONF="/etc/lightdm/lightdm.conf"
if [ -f "$LIGHTDM_CONF" ]; then
    sudo sed -i "s/#autologin-user=.*/autologin-user=$USER/" "$LIGHTDM_CONF"
    sudo sed -i "s/#autologin-user-timeout=.*/autologin-user-timeout=0/" "$LIGHTDM_CONF"
fi

# GDM auto-login (if using GDM)
GDM_CONF="/etc/gdm3/custom.conf"
if [ -f "$GDM_CONF" ]; then
    sudo sed -i "s/#  AutomaticLoginEnable =.*/AutomaticLoginEnable = true/" "$GDM_CONF"
    sudo sed -i "s/#  AutomaticLogin =.*/AutomaticLogin = $USER/" "$GDM_CONF"
fi

echo ""
echo "✅ Boot splash installed!"
echo ""
echo "What the student will see on boot:"
echo "  1. Black screen → VIRON logo with pulsing glow"
echo "  2. 'VIRON AI Tutor • Initializing...' text"
echo "  3. Loading animation"
echo "  4. Auto-login (no Ubuntu desktop visible)"
echo "  5. VIRON face launches in fullscreen kiosk"
echo ""
echo "🔄 Reboot to test: sudo reboot"
