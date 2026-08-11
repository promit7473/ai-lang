# Deployment Instructions

## 1. Create GitHub Repository

Run this in your terminal:

```bash
# Authenticate with GitHub
"C:\Program Files\GitHub CLI\gh.exe" auth login

# Create the repository
"C:\Program Files\GitHub CLI\gh.exe" repo create ai-lang --public --description "A high-level programming language for guiding LLMs in robotics and AI agent tasks"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/ai-lang.git
git branch -M main
git push -u origin main
```

## 2. Enable GitHub Pages

1. Go to https://github.com/YOUR_USERNAME/ai-lang/settings/pages
2. Source: Deploy from branch
3. Branch: `gh-pages` / `/ (root)`
4. Click Save

Your site will be live at: `https://YOUR_USERNAME.github.io/ai-lang/`

## 3. (Alternative) Deploy docs manually

```bash
# Install ghp-import
pip install ghp-import

# Deploy to GitHub Pages branch
ghp-import -n -p -f site
```

## 4. View locally

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.
