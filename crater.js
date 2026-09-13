(() => {
  "use strict";

  const DEG = Math.PI / 180;
  const RAD = 180 / Math.PI;

  const body = document.getElementById("pointsBody");
  const status = document.getElementById("status");
  const planetRadius = document.getElementById("planetRadius");
  const radiusUnit = document.getElementById("radiusUnit");
  const canvas = document.getElementById("plot");
  const ctx = canvas.getContext("2d");

  const centerLat = document.getElementById("centerLat");
  const centerLon = document.getElementById("centerLon");
  const angularRadius = document.getElementById("angularRadius");
  const surfaceRadius = document.getElementById("surfaceRadius");
  const meanError = document.getElementById("meanError");
  const maxError = document.getElementById("maxError");
  const rmsError = document.getElementById("rmsError");

  let pointCounter = 0;

  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function normalizeLon(lon) {
    let x = ((lon + 180) % 360 + 360) % 360 - 180;
    if (x === -180 && lon > 0) x = 180;
    return x;
  }

  function latLonToVector(latDeg, lonDeg) {
    const lat = latDeg * DEG;
    const lon = lonDeg * DEG;
    const c = Math.cos(lat);
    return [c * Math.cos(lon), c * Math.sin(lon), Math.sin(lat)];
  }

  function dot(a, b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  }

  function norm(v) {
    return Math.hypot(v[0], v[1], v[2]);
  }

  function normalize(v) {
    const n = norm(v);
    return [v[0]/n, v[1]/n, v[2]/n];
  }

  function add(a, b) {
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
  }

  function scale(a, s) {
    return [a[0]*s, a[1]*s, a[2]*s];
  }

  function subtract(a, b) {
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  }

  function cross(a, b) {
    return [
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    ];
  }

  // Symmetric 3x3 Jacobi eigensolver.
  // Returns eigenvalues/eigenvectors for a symmetric matrix.
  function jacobiEigenSymmetric(A) {
    const a = A.map(row => row.slice());
    const v = [
      [1,0,0],
      [0,1,0],
      [0,0,1]
    ];

    for (let iter = 0; iter < 60; iter++) {
      let p = 0, q = 1;
      let max = Math.abs(a[0][1]);

      if (Math.abs(a[0][2]) > max) { max = Math.abs(a[0][2]); p = 0; q = 2; }
      if (Math.abs(a[1][2]) > max) { max = Math.abs(a[1][2]); p = 1; q = 2; }

      if (max < 1e-14) break;

      const theta = 0.5 * Math.atan2(2*a[p][q], a[q][q] - a[p][p]);
      const c = Math.cos(theta);
      const s = Math.sin(theta);

      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c*apk - s*aqk;
        a[q][k] = s*apk + c*aqk;
      }

      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c*akp - s*akq;
        a[k][q] = s*akp + c*akq;
      }

      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c*vkp - s*vkq;
        v[k][q] = s*vkp + c*vkq;
      }
    }

    return {
      values: [a[0][0], a[1][1], a[2][2]],
      vectors: [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]]
      ]
    };
  }

  // Initial spherical-circle fit:
  // Find the direction whose dot products with the points have minimum variance.
  function initialFit(points) {
    const centroid = points.reduce(add, [0,0,0]);
    const mean = scale(centroid, 1 / points.length);

    const S = [
      [0,0,0],
      [0,0,0],
      [0,0,0]
    ];

    for (const p of points) {
      const q = subtract(p, mean);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          S[r][c] += q[r] * q[c];
        }
      }
    }

    const eig = jacobiEigenSymmetric(S);
    let idx = 0;
    for (let i = 1; i < 3; i++) {
      if (eig.values[i] < eig.values[idx]) idx = i;
    }

    let center = normalize(eig.vectors[idx]);

    // Pick the hemisphere containing the data. This gives the small circle
    // when the crater is a normal local feature rather than a near-great-circle.
    if (dot(center, mean) < 0) center = scale(center, -1);

    const k = points.reduce((sum, p) => sum + dot(center, p), 0) / points.length;
    const alpha = Math.acos(clamp(k, -1, 1));

    return { center, alpha };
  }

  function angularDistance(a, b) {
    return Math.acos(clamp(dot(a,b), -1, 1));
  }

  // Refine the center on the unit sphere using numerical Gauss-Newton.
  // Residual = angularDistance(center, point) - radius.
  function refineFit(points, initialCenter, initialAlpha) {
    let c = normalize(initialCenter);
    let alpha = initialAlpha;

    const eps = 1e-6;

    for (let iter = 0; iter < 30; iter++) {
      const lat = Math.asin(clamp(c[2], -1, 1));
      const lon = Math.atan2(c[1], c[0]);

      // Tangent basis at c: east and north.
      let east = [-Math.sin(lon), Math.cos(lon), 0];
      let north = [
        -Math.sin(lat)*Math.cos(lon),
        -Math.sin(lat)*Math.sin(lon),
        Math.cos(lat)
      ];

      const residuals = [];
      const J = [];

      for (const p of points) {
        const d = angularDistance(c, p);
        residuals.push(d - alpha);

        // Numerical derivatives with respect to two tangent-plane
        // center movements and radius.
        const cp = normalize(add(c, scale(east, eps)));
        const cm = normalize(subtract(c, scale(east, eps)));
        const np = normalize(add(c, scale(north, eps)));
        const nm = normalize(subtract(c, scale(north, eps)));

        const de = (angularDistance(cp, p) - angularDistance(cm, p)) / (2*eps);
        const dn = (angularDistance(np, p) - angularDistance(nm, p)) / (2*eps);

        J.push([de, dn, -1]);
      }

      // Solve normal equations (J^T J) dx = -J^T r.
      const A = [[0,0,0],[0,0,0],[0,0,0]];
      const b = [0,0,0];

      for (let i = 0; i < points.length; i++) {
        for (let r = 0; r < 3; r++) {
          b[r] += J[i][r] * residuals[i];
          for (let col = 0; col < 3; col++) {
            A[r][col] += J[i][r] * J[i][col];
          }
        }
      }

      const dx = solve3x3(A, scale(b, -1));
      if (!dx) break;

      const stepSize = Math.hypot(dx[0], dx[1], dx[2]);
      if (stepSize < 1e-11) break;

      c = normalize(add(c, add(scale(east, dx[0]), scale(north, dx[1]))));
      alpha = Math.max(0, Math.min(Math.PI, alpha + dx[2]));
    }

    return { center: c, alpha };
  }

  function solve3x3(A, b) {
    const m = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < 3; col++) {
      let pivot = col;
      for (let r = col + 1; r < 3; r++) {
        if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
      }

      if (Math.abs(m[pivot][col]) < 1e-14) return null;

      [m[col], m[pivot]] = [m[pivot], m[col]];

      for (let r = col + 1; r < 3; r++) {
        const f = m[r][col] / m[col][col];
        for (let k = col; k < 4; k++) m[r][k] -= f * m[col][k];
      }
    }

    const x = [0,0,0];
    for (let r = 2; r >= 0; r--) {
      let s = m[r][3];
      for (let c = r + 1; c < 3; c++) s -= m[r][c] * x[c];
      x[r] = s / m[r][r];
    }
    return x;
  }

  function vectorToLatLon(v) {
    const lat = Math.atan2(v[2], Math.hypot(v[0], v[1])) * RAD;
    const lon = normalizeLon(Math.atan2(v[1], v[0]) * RAD);
    return { lat, lon };
  }

  function getUnitScale() {
    const unit = radiusUnit.value;
    if (unit === "m") return { kmPerUnit: 0.001, label: "m", toUnit: x => x * 1000 };
    if (unit === "mi") return { kmPerUnit: 1.609344, label: "mi", toUnit: x => x / 1.609344 };
    return { kmPerUnit: 1, label: "km", toUnit: x => x };
  }

  function formatDegrees(x) {
    return `${x.toFixed(6)}°`;
  }

  function formatDistance(x) {
    const ax = Math.abs(x);
    if (ax < 0.001) return `${x.toExponential(3)}`;
    if (ax < 1) return `${x.toFixed(4)}`;
    if (ax < 100) return `${x.toFixed(2)}`;
    return `${x.toFixed(1)}`;
  }

  function getInputPoints() {
    const rows = [...body.querySelectorAll("tr")];
    const points = [];

    for (const row of rows) {
      const lat = Number(row.querySelector(".lat").value);
      const lon = Number(row.querySelector(".lon").value);

      if (row.querySelector(".lat").value.trim() === "" ||
          row.querySelector(".lon").value.trim() === "") {
        continue;
      }

      if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
        throw new Error("All entered coordinates must be valid numbers.");
      }
      if (lat < -90 || lat > 90) {
        throw new Error("Latitude must be between −90° and +90°.");
      }
      if (lon < -180 || lon > 180) {
        throw new Error("Longitude must be between −180° and +180°.");
      }

      points.push({
        lat,
        lon,
        vector: latLonToVector(lat, lon)
      });
    }

    return points;
  }

  function calculate() {
    try {
      const input = getInputPoints();

      if (input.length < 3) {
        setResultDashes();
        status.textContent = `Enter at least 3 valid points (${input.length} entered).`;
        drawEmpty();
        return;
      }

      const vectors = input.map(p => p.vector);
      const initial = initialFit(vectors);
      const fit = refineFit(vectors, initial.center, initial.alpha);
      const ll = vectorToLatLon(fit.center);

      const distances = vectors.map(p => angularDistance(fit.center, p));
      const errors = distances.map(d => d - fit.alpha);
      const absErrors = errors.map(Math.abs);
      const meanAbs = absErrors.reduce((a,b) => a+b, 0) / errors.length;
      const maxAbs = Math.max(...absErrors);
      const rms = Math.sqrt(errors.reduce((s,e) => s + e*e, 0) / errors.length);

      const radiusValue = Number(planetRadius.value);
      const scaleInfo = getUnitScale();

      if (!Number.isFinite(radiusValue) || radiusValue <= 0) {
        throw new Error("Planet radius must be greater than zero.");
      }

      const planetRadiusKm = radiusValue * scaleInfo.kmPerUnit;
      const surfaceRadiusKm = planetRadiusKm * fit.alpha;
      const meanErrorKm = planetRadiusKm * meanAbs;
      const maxErrorKm = planetRadiusKm * maxAbs;
      const rmsErrorKm = planetRadiusKm * rms;

      centerLat.textContent = formatDegrees(ll.lat);
      centerLon.textContent = formatDegrees(ll.lon);
      angularRadius.textContent = `${(fit.alpha * RAD).toFixed(6)}°`;
      surfaceRadius.textContent = `${formatDistance(scaleInfo.toUnit(surfaceRadiusKm))} ${scaleInfo.label}`;
      meanError.textContent = `${formatDistance(scaleInfo.toUnit(meanErrorKm))} ${scaleInfo.label}`;
      maxError.textContent = `${formatDistance(scaleInfo.toUnit(maxErrorKm))} ${scaleInfo.label}`;
      rmsError.textContent = `${formatDistance(scaleInfo.toUnit(rmsErrorKm))} ${scaleInfo.label}`;

      status.textContent = `Fit calculated from ${input.length} points.`;
      drawPlot(input, fit.center, fit.alpha);
    } catch (err) {
      setResultDashes();
      status.textContent = err.message;
      drawEmpty();
    }
  }

  function setResultDashes() {
    for (const el of [centerLat, centerLon, angularRadius, surfaceRadius, meanError, maxError, rmsError]) {
      el.textContent = "—";
    }
  }

  function addPoint(lat = "", lon = "") {
    pointCounter++;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="number">${pointCounter}</td>
      <td><input class="coordinate lat" type="number" min="-90" max="90" step="any" placeholder="e.g. 12.345678" value="${lat}"></td>
      <td><input class="coordinate lon" type="number" min="-180" max="180" step="any" placeholder="e.g. 45.678901" value="${lon}"></td>
      <td><button type="button" class="delete" title="Remove point">Remove</button></td>
    `;

    tr.querySelectorAll("input").forEach(input => input.addEventListener("input", calculate));
    tr.querySelector(".delete").addEventListener("click", () => {
      tr.remove();
      renumber();
      calculate();
    });

    body.appendChild(tr);
    calculate();
  }

  function renumber() {
    [...body.querySelectorAll("tr")].forEach((tr, i) => {
      tr.querySelector(".number").textContent = i + 1;
    });
  }

  function loadExample() {
    body.innerHTML = "";
    pointCounter = 0;

    // Synthetic crater centered near 25°N, 40°E with a radius of about 3°.
    const example = [
      [28.00, 40.00],
      [26.93, 42.32],
      [23.48, 42.06],
      [22.03, 39.10],
      [24.20, 37.17],
      [27.18, 37.86]
    ];

    example.forEach(([lat, lon]) => addPoint(lat, lon));
    calculate();
  }

  function clearPoints() {
    body.innerHTML = "";
    pointCounter = 0;
    addPoint();
    addPoint();
    addPoint();
    calculate();
  }

  function destination(center, bearing, angularDistanceValue) {
    // Useful for drawing the fitted circle.
    const lat1 = Math.asin(center[2]);
    const lon1 = Math.atan2(center[1], center[0]);

    const sinLat1 = Math.sin(lat1);
    const cosLat1 = Math.cos(lat1);
    const sinD = Math.sin(angularDistanceValue);
    const cosD = Math.cos(angularDistanceValue);

    const lat = Math.asin(clamp(
      sinLat1*cosD + cosLat1*sinD*Math.cos(bearing), -1, 1
    ));

    const lon = lon1 + Math.atan2(
      Math.sin(bearing)*sinD*cosLat1,
      cosD - sinLat1*Math.sin(lat)
    );

    return latLonToVector(lat * RAD, lon * RAD);
  }

  function project(point, center, basisEast, basisNorth) {
    const d = angularDistance(center, point);
    if (d < 1e-12) return { x: 0, y: 0 };

    const sinD = Math.sin(d);
    const scaleFactor = d / sinD;

    return {
      x: dot(point, basisEast) * scaleFactor,
      y: dot(point, basisNorth) * scaleFactor
    };
  }

  function drawPlot(input, center, alpha) {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const lat = Math.asin(clamp(center[2], -1, 1));
    const lon = Math.atan2(center[1], center[0]);

    const east = [-Math.sin(lon), Math.cos(lon), 0];
    const north = [
      -Math.sin(lat)*Math.cos(lon),
      -Math.sin(lat)*Math.sin(lon),
      Math.cos(lat)
    ];

    const projected = input.map(p => project(p.vector, center, east, north));
    const circle = [];
    for (let i = 0; i <= 360; i++) {
      circle.push(project(destination(center, i*DEG, alpha), center, east, north));
    }

    let maxR = alpha;
    for (const p of projected) maxR = Math.max(maxR, Math.hypot(p.x, p.y));
    maxR = Math.max(maxR * 1.18, 1e-4);

    const cx = w / 2, cy = h / 2;
    const radiusPx = Math.min(w, h) * 0.40;
    const pxPerRad = radiusPx / maxR;

    // Grid rings.
    ctx.save();
    ctx.translate(cx, cy);

    ctx.lineWidth = 1;
    ctx.strokeStyle = "#d9dee7";
    ctx.fillStyle = "#697386";
    ctx.font = "13px system-ui, sans-serif";

    for (let fraction = 0.25; fraction <= 1.0; fraction += 0.25) {
      ctx.beginPath();
      ctx.arc(0, 0, radiusPx*fraction, 0, Math.PI*2);
      ctx.stroke();

      const label = `${(maxR*fraction*RAD).toFixed(1)}°`;
      ctx.fillText(label, 7, -radiusPx*fraction - 5);
    }

    // Fitted circle.
    ctx.beginPath();
    circle.forEach((p, i) => {
      const x = p.x * pxPerRad, y = -p.y * pxPerRad;
      if (i === 0) ctx.moveTo(x,y);
      else ctx.lineTo(x,y);
    });
    ctx.strokeStyle = "#263a70";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Crater center.
    ctx.beginPath();
    ctx.arc(0, 0, 6, 0, Math.PI*2);
    ctx.fillStyle = "#263a70";
    ctx.fill();

    // Points.
    projected.forEach((p, i) => {
      const x = p.x * pxPerRad;
      const y = -p.y * pxPerRad;

      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI*2);
      ctx.fillStyle = "#9a3341";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#273149";
      ctx.font = "bold 12px system-ui, sans-serif";
      ctx.fillText(String(i+1), x + 9, y - 8);
    });

    ctx.restore();

    ctx.fillStyle = "#273149";
    ctx.font = "14px system-ui, sans-serif";
    ctx.fillText("● center", 18, 26);
    ctx.fillText("● edge points", 18, 49);
  }

  function drawEmpty() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#697386";
    ctx.font = "16px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Enter at least three valid points to display the fitted crater.", w/2, h/2);
    ctx.textAlign = "left";
  }

  document.getElementById("addPoint").addEventListener("click", () => addPoint());
  document.getElementById("example").addEventListener("click", loadExample);
  document.getElementById("clear").addEventListener("click", clearPoints);
  planetRadius.addEventListener("input", calculate);
  radiusUnit.addEventListener("change", calculate);

  // Start with 3 empty rows.
  addPoint();
  addPoint();
  addPoint();
  drawEmpty();
})();
