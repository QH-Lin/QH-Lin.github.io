# Lin Qihong's Log

Personal learning-notes blog by 林琦宏 (Lin Qihong), built with
[Hugo](https://gohugo.io/) and the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

Website: https://qh-lin.github.io/

## Local development

PaperMod is tracked as a Git submodule. Hugo Extended is installed locally in
`.tools/` so it does not modify the system environment.

```bash
git clone --recurse-submodules https://github.com/QH-Lin/QH-Lin.github.io.git
cd QH-Lin.github.io
./scripts/setup-hugo.sh
.tools/hugo server --buildDrafts
```

Open http://localhost:1313/ to preview the site.

## Writing

Create a draft:

```bash
.tools/hugo new content posts/my-post.md
```

Drafts are hidden from production until their front matter contains:

```yaml
draft: false
```

MathJax is enabled for posts with `math: true`; write inline formulas with
`$...$` and display formulas with `$$...$$`.

Build the production site:

```bash
.tools/hugo --gc --minify
```

The generated site is written to `public/`. GitHub Actions publishes it to
GitHub Pages after changes reach `main`.
