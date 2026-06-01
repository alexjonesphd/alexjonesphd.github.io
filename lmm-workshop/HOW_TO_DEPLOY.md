# Deploying the LMM workshop page

The folder `lmm-workshop/` contains the Quarto source for a new page plus all assets.

## Step by step

1. Copy the **entire `lmm-workshop/` folder** into the root of your Quarto project (the source for `alexjonesphd.github.io`).
2. The folder layout should now look like:

   ```
   <site-root>/
     teaching.qmd
     publications.qmd
     ...
     lmm-workshop/
       lmm-workshop.qmd
       files/
         LMM_morning_lecture.pptx
         LMM_Workshop_Morning_Worksheet.html
         practical_slides.html
         low_n_missing.csv
         face_rt_exercise_worksheet.html
         exercise_data.csv
   ```

3. Build the site (`quarto render` locally, or push and let your Actions workflow do it).
4. The page will be live at:
   **https://alexjonesphd.github.io/lmm-workshop/lmm-workshop.html**

That's the URL you share with attendees.

### Make sure the asset files get published

Quarto only copies files into `_site/` if it knows about them. The `.qmd` already references each asset via a relative link, so Quarto's link-checker will pick them up and copy the `files/` folder across. If you ever see broken links after a build, add this to your `_quarto.yml` `project:` block:

```yaml
project:
  resources:
    - "lmm-workshop/files/**"
```

That tells Quarto to always copy the assets, regardless of how they're referenced.

### Optional: prettier URL

If you'd like the link to be just `https://alexjonesphd.github.io/lmm-workshop/`, rename `lmm-workshop.qmd` to `index.qmd`. Same content, shorter URL.

### Optional: link from the Teaching page

Edit `teaching.qmd` and add a section, e.g.:

```markdown
### Linear Mixed Models Workshop
[alexjonesphd.github.io/lmm-workshop](lmm-workshop/lmm-workshop.html)

A one-day workshop on linear mixed-effects models in Python — slides, worksheets, datasets, exercise.
```

## A note on the buttons

The download/open buttons in the `.qmd` use Quarto's standard Bootstrap classes (`.btn .btn-primary`, `.btn .btn-secondary`) plus the HTML5 `download` and `target="_blank"` attributes. Nothing custom — they'll pick up your site's theme colours automatically. If you don't like blue/grey, change the classes (`.btn-success`, `.btn-outline-primary`, etc.).

## What's in `files/`

| File | Used in |
| --- | --- |
| `LMM_morning_lecture.pptx` | Morning lecture slides |
| `LMM_Workshop_Morning_Worksheet.html` | Morning worksheet (follow-along) |
| `practical_slides.html` | Afternoon practical slides |
| `low_n_missing.csv` | Dataset for the afternoon practical |
| `face_rt_exercise_worksheet.html` | Take-home exercise handout |
| `exercise_data.csv` | Dataset for the exercise |
