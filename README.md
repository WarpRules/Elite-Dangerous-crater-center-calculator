# Crater Center Calculator

A client-side web utility for estimating the center of a circular crater from latitude/longitude points on a spherical planet.

## Files

- `index.html` — page structure
- `style.css` — styling
- `crater.js` — spherical least-squares fitting and visualization

## Running locally

Open `index.html` in a modern web browser. No server or build step is required.

## GitHub Pages

1. Create a GitHub repository.
2. Upload the three web files.
3. Commit/push them to the repository.
4. In GitHub, open **Settings → Pages**.
5. Select deployment from the repository's main branch and root folder.
6. GitHub will provide the Pages URL.

## Mathematics

Coordinates are converted from latitude/longitude to 3D unit vectors. A spherical circle is represented by

    c · p = cos(alpha)

where `c` is the unit vector pointing toward the crater center and `alpha` is its angular radius.

An eigenvector of the centered 3D scatter matrix provides an initial best-fit center. The implementation then refines the center and radius by minimizing the squared differences between the points' angular distances from the center and the fitted radius.

All calculations are performed locally in the browser.
