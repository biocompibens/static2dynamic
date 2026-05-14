// Edit this object only: the rest of the file just renders the page.
const siteData = {
  title: "Static2Dynamic",
  subtitle:
    "Supplementary material with pseudotimes, ground truths and generated videos.",
  intro: [
    "Static2Dynamic reconstructs unseen continuous dynamics from time-unpaired static images. It achieves this in 3 stages: 1. estimate a continuous pseudotime for each image 2. learn a pseudotime-conditioned image diffusion model 3. generate temporally coherent videos initialized from real samples",
    "We validate our method on experimental microscopy *video* datasets where ground truths are available, then apply Static2Dynamic to biological processes where only cross-sectional data is available, demonstrating its wide applicability.",
  ],
  datasets: [
    {
      name: "Biotine",
      size: "large",
      patchGrid: { rows: 8, cols: 8 },
      mediaNote: "Light grid lines mark the patch boundaries used for generation.",
      plot: "biotine/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "An in-house time lapse assay of human A549 lung cancer cells treated with biotine and imaged by confocal microscopy at the Curie Institute. Cells are marked with GFP for biotin effect, Rhodamin for membrane, and NucLight for nuclei. The videos shown below are collages of individual image patches.",
      comparisons: [
        {
          gt: { src: "biotine/display_vids/ground_truth_01_10s.mp4" },
          gen: { src: "biotine/display_vids/generated_01_10s.mp4" },
        },
        {
          gt: { src: "biotine/display_vids/ground_truth_02_10s.mp4" },
          gen: { src: "biotine/display_vids/generated_02_10s.mp4" },
        },
      ],
    },
    {
      name: "ChromaLive",
      size: "large",
      patchGrid: { rows: 5, cols: 5 },
      mediaNote: "Light grid lines mark the patch boundaries used for generation.",
      plot: "chromalive/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "A time lapse assay from [Lippincott et al. 2025]{lippincott2025}. HeLa cells were exposed to staurosporine, processed with the Live Cell Painting assay (ChromaLIVE), and imaged every 30 minutes for six hours using spinning-disk confocal microscopy. The videos shown below are collages of individual image patches.",
      comparisons: [
        {
          gt: { src: "chromalive/display_vids/ground_truth_01_10s.mp4" },
          gen: { src: "chromalive/display_vids/generated_01_10s.mp4" },
        },
        {
          gt: { src: "chromalive/display_vids/ground_truth_02_10s.mp4" },
          gen: { src: "chromalive/display_vids/generated_02_10s.mp4" },
        },
      ],
    },
    {
      name: "Cell cycle",
      size: "small",
      plot: "cell_cycle/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "An imaging flow cytometry dataset of Jurkat cells from [Blasi et al. 2016]{blasi2016}. The discrete classes are annotated cell cycle phases. We use only the brightfield channel and reprocess the raw single-cell crops by standardizing square crops, filling the background, and aligning intensity.",
      comparisons: [
        {
          gt: {
            src: "cell_cycle/gt_vids_display/ground_truth_grid_01.png",
            labels: [
              "G1",
              "S",
              "G2",
              "Prophase",
              "Metaphase",
              "Anaphase",
              "Telophase",
            ],
          },
          gen: {
            src: "cell_cycle/gen_vids/extracted/trajectories_cell_0_0.mp4",
          },
        },
        {
          gt: {
            src: "cell_cycle/gt_vids_display/ground_truth_grid_02.png",
            labels: [
              "G1",
              "S",
              "G2",
              "Prophase",
              "Metaphase",
              "Anaphase",
              "Telophase",
            ],
          },
          gen: {
            src: "cell_cycle/gen_vids/extracted/trajectories_cell_1_0.mp4",
          },
        },
      ],
    },
    {
      name: "Ependymal single-cell",
      size: "small",
      plot: "ependymal/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "Images from [Bankole et al. 2025]{bankole2025} on ependymal cells. Mouse lateral ependymal wall tissues were collected across postnatal stages, stained for cell junctions, centrioles, and deuterosomes, projected from 3D stacks to 2D ventricular surface images ([Shihavuddin et al. 2017]{shihavuddin2017}), then segmented into individual cell images.",
      comparisons: [
        {
          gt: {
            src: "ependymal/gt_imgs/ground_truth_grid_01.png",
            labels: ["stem", "halo", "flower", "individualization", "crescent", "ependymal"],
          },
          gen: { src: "ependymal/gen_vids/extracted/trajectories_cell_2_2.mp4" },
        },
        {
          gt: {
            src: "ependymal/gt_imgs/ground_truth_grid_02.png",
            labels: ["stem", "halo", "flower", "individualization", "crescent", "ependymal"],
          },
          gen: { src: "ependymal/gen_vids/extracted/trajectories_cell_2_3.mp4" },
        },
      ],
    },
    {
      name: "NASH steatosis",
      size: "small",
      plot: "NASH/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "A preprocessed dataset from [Heinemann et al. 2019]{heinemann2019} with mouse or rat liver tissue sections affected by non-alcoholic fatty liver disease, stained with Masson's trichrome, and annotated for four steatosis stages corresponding to the Kleiner score ([Kleiner et al. 2005]{kleiner2005}).",
      comparisons: [
        {
          gt: {
            src: "NASH/gt_vids/ground_truth_grid_01.png",
            labels: ["0", "1", "2", "3"],
          },
          gen: { src: "NASH/gen_vids/extracted/trajectories_cell_0_0.mp4" },
        },
        {
          gt: {
            src: "NASH/gt_vids/ground_truth_grid_02.png",
            labels: ["0", "1", "2", "3"],
          },
          gen: { src: "NASH/gen_vids/extracted/trajectories_cell_1_0.mp4" },
        },
      ],
    },
    {
      name: "Retinopathy",
      size: "small",
      plot: "retino/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "A repurposed Kaggle classification dataset ([Diabetic Retinopathy Detection 2015]{retinopathy2015}) of high-resolution retina fundus images with variable imaging conditions and subtle phenotypic differences. Images are annotated on five diabetic retinopathy scales: no DR, mild, moderate, severe, and proliferative DR.",
      comparisons: [
        {
          gt: {
            src: "retino/gt_imgs/ground_truth_grid_01.png",
            labels: ["0", "1", "2", "3", "4"],
          },
          gen: { src: "retino/gen_vids/extracted/trajectories_cell_3_0.mp4" },
        },
        {
          gt: {
            src: "retino/gt_imgs/ground_truth_grid_02.png",
            labels: ["0", "1", "2", "3", "4"],
          },
          gen: { src: "retino/gen_vids/extracted/trajectories_cell_3_2.mp4" },
        },
      ],
    },
    {
      name: "Docetaxel on MCF-7",
      size: "small",
      plot: "docetaxel/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "A Docetaxel subset of the BBBC021 phenotypic profiling dataset from [Caie et al. 2010]{caie2010}, available from the Broad Bioimage Benchmark Collection ([Ljosa et al. 2012]{ljosa2012}). MCF-7 breast cancer cells are treated across eight concentrations and labeled for DNA, F-actin, and beta-tubulin.",
      comparisons: [
        {
          gt: {
            src: "docetaxel/gt_imgs/ground_truth_grid_01.png",
            labels: ["0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"],
          },
          gen: { src: "docetaxel/gen_vids/extracted/trajectories_cell_1_1.mp4" },
        },
        {
          gt: {
            src: "docetaxel/gt_imgs/ground_truth_grid_02.png",
            labels: ["0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"],
          },
          gen: { src: "docetaxel/gen_vids/extracted/trajectories_cell_0_0.mp4" },
        },
      ],
    },
    {
      name: "Nocodazole on MCF-7",
      size: "small",
      plot: "nocodazole/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "A Nocodazole subset of the BBBC021 phenotypic profiling dataset from [Caie et al. 2010]{caie2010}, available from the Broad Bioimage Benchmark Collection ([Ljosa et al. 2012]{ljosa2012}). MCF-7 breast cancer cells are treated across eight concentrations and labeled for DNA, F-actin, and beta-tubulin.",
      comparisons: [
        {
          gt: {
            src: "nocodazole/gt_imgs/ground_truth_grid_01.png",
            labels: ["DMSO", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0", "3.0"],
          },
          gen: { src: "nocodazole/gen_vids/extracted/trajectories_cell_0_3.mp4" },
        },
        {
          gt: {
            src: "nocodazole/gt_imgs/ground_truth_grid_02.png",
            labels: ["DMSO", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0", "3.0"],
          },
          gen: { src: "nocodazole/gen_vids/extracted/trajectories_cell_0_1.mp4" },
        },
      ],
    },
  ],
};

const imageExtensions = new Set([
  "png",
  "jpg",
  "jpeg",
  "webp",
  "gif",
  "svg",
  "avif",
]);
const videoExtensions = new Set(["mp4", "mov", "webm", "ogg", "avi", "m4v"]);

const escapeHtml = (value = "") =>
  value.replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[char];
  });

const citations = window.siteCitations || {};

const richText = (value = "") =>
  escapeHtml(value)
    .replace(/\[([^\]]+)\]\{([a-zA-Z0-9_-]+)\}/g, (match, label, id) => {
      const citation = citations[id];
      if (!citation?.url) {
        return label;
      }
      return `<a class="citation-link" href="${encodeURI(citation.url)}" target="_blank" rel="noreferrer">${escapeHtml(citation.label || label)}</a>`;
    })
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");

const renderIntroParagraph = (paragraph) => {
  const stages = paragraph.match(
    /^(.*?:)\s*1\.\s*(.*?)\s*2\.\s*(.*?)\s*3\.\s*(.*)$/,
  );
  if (!stages) {
    return `<p>${richText(paragraph)}</p>`;
  }

  const [, lead, ...items] = stages;
  return `
    <p>${richText(lead)}</p>
    <ol class="intro-steps">
      ${items.map((item) => `<li>${richText(item)}</li>`).join("")}
    </ol>
  `;
};

const fileStem = (path) =>
  path
    .split("/")
    .pop()
    .replace(/\.[^.]+$/, "");
const titleFromPath = (path) =>
  fileStem(path).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

const slugify = (value) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

const mediaKind = (item) => {
  if (item.type === "image" || item.type === "video") {
    return item.type;
  }

  const extension = item.src.split(".").pop().toLowerCase();

  if (imageExtensions.has(extension)) {
    return "image";
  }

  if (videoExtensions.has(extension)) {
    return "video";
  }

  return "video";
};

const gridLines = (cols, rows) => {
  const vertical = Array.from({ length: Math.max(cols - 1, 0) }, (_, index) => {
    const x = index + 1;
    return `<line x1="${x}" y1="0" x2="${x}" y2="${rows}"></line>`;
  }).join("");
  const horizontal = Array.from({ length: Math.max(rows - 1, 0) }, (_, index) => {
    const y = index + 1;
    return `<line x1="0" y1="${y}" x2="${cols}" y2="${y}"></line>`;
  }).join("");
  return `<rect x="0" y="0" width="${cols}" height="${rows}"></rect>${vertical}${horizontal}`;
};

const patchGridSvg = (grid, extraClass = "") => {
  const cols = Number(grid?.cols);
  const rows = Number(grid?.rows);
  if (!Number.isFinite(cols) || !Number.isFinite(rows) || cols < 1 || rows < 1) {
    return "";
  }
  return `<svg class="patch-grid${extraClass}" viewBox="0 0 ${cols} ${rows}" preserveAspectRatio="none" aria-hidden="true">${gridLines(cols, rows)}</svg>`;
};

const mediaCard = (item, label, datasetSize = "small", patchGrid = null) => {
  const source = encodeURI(item.src);
  const kind = mediaKind(item);
  const isImage = kind === "image";
  const isLargeDataset = datasetSize === "large";
  const grid = !isImage && patchGrid ? patchGrid : null;
  const gridData = grid
    ? ` data-grid-cols="${grid.cols}" data-grid-rows="${grid.rows}"`
    : "";
  const gridOverlay = grid ? patchGridSvg(grid) : "";
  const showVideoState = !isImage && label === "Generated";
  const videoStateToggle = showVideoState
    ? `<button class="video-state-toggle" type="button" data-video-state="paused" aria-label="Play generated video">
         <span class="video-state-icon" aria-hidden="true">
           <span class="video-state-play"></span>
           <span class="video-state-pause"></span>
         </span>
       </button>`
    : "";
  const labels = Array.isArray(item.labels) ? item.labels : [];
  const zoomLabels = labels.length
    ? ` data-zoom-labels="${escapeHtml(JSON.stringify(labels))}"`
    : "";
  const zoomTitle = ` data-zoom-title="${escapeHtml(label)}"`;
  const labelTicks = labels.length
    ? `<div class="series-labels" style="--series-count: ${labels.length}">
        ${labels.map((seriesLabel) => `<span>${escapeHtml(seriesLabel)}</span>`).join("")}
      </div>`
    : "";
  const media = isImage
    ? `<button class="image-zoom-trigger" type="button" data-zoom-src="${source}"${zoomTitle}${zoomLabels} aria-label="Zoom ${escapeHtml(label)}">
           <img src="${source}" alt="${escapeHtml(label)}" loading="lazy">
         </button>`
    : `<span class="video-frame${grid ? " video-frame-grid" : ""}">
           <video controls playsinline muted preload="none" data-sync-video data-viewport-autoplay>
             <source src="${source}">
             Your browser does not support the video tag.
           </video>
           ${gridOverlay}
         </span>
         <button class="video-zoom-trigger" type="button" data-zoom-src="${source}"${zoomTitle}${gridData} data-zoom-size="${isLargeDataset ? "large" : "small"}" aria-label="Zoom ${escapeHtml(label)}">Zoom</button>`;

  return `
    <article class="media-card ${isImage ? "media-card-image" : "media-card-video"}">
      <div class="media-label-row">
        <p class="media-label">${escapeHtml(label)}</p>
        ${videoStateToggle}
      </div>
      <div class="media-shell">${media}</div>
      ${labelTicks}
    </article>
  `;
};

const plotEmbeds = (dataset) => {
  const plots = Array.isArray(dataset.plots)
    ? dataset.plots
    : dataset.plot
      ? [dataset.plot]
      : [];
  if (!plots.length) {
    return "";
  }
  return plots
    .map(
      (plot) => `
    <div class="plot-embed">
      <iframe src="${encodeURI(plot)}" loading="lazy" title="Dataset embedding plot"></iframe>
    </div>
  `,
    )
    .join("");
};

const normalizeComparisons = (dataset) => {
  if (Array.isArray(dataset.comparisons)) {
    return dataset.comparisons;
  }

  const gtItems = Array.isArray(dataset.groundTruth) ? dataset.groundTruth : [];
  const genItems = Array.isArray(dataset.generated) ? dataset.generated : [];
  const count = Math.max(gtItems.length, genItems.length);
  return Array.from({ length: count }, (_, index) => ({
    gt: gtItems[index],
    gen: genItems[index],
  }));
};

const comparisonRow = (comparison, datasetSize, patchGrid = null) => {
  const gt = comparison.gt;
  const gen = comparison.gen;
  const gtLabel =
    gt && mediaKind(gt) === "video" ? "Ground truth" : "Ground truths";
  return `
    <article class="comparison-row">
      <div class="comparison-media">
        ${gt ? mediaCard(gt, gtLabel, datasetSize, patchGrid) : `<div class="empty-note">No ground truth sample.</div>`}
        ${gen ? mediaCard(gen, "Generated", datasetSize, patchGrid) : `<div class="empty-note">No generated sample.</div>`}
      </div>
    </article>
  `;
};

const mediaNote = (text) =>
  text ? `<p class="media-note">${richText(text)}</p>` : "";

const comparisonList = (comparisons, size, patchGrid = null) => `
  <div class="comparison-list">
    ${
      comparisons.length
        ? comparisons
            .map((comparison) => comparisonRow(comparison, size, patchGrid))
            .join("")
        : `<div class="empty-note">No media added yet.</div>`
    }
  </div>
`;

const datasetBlocks = (dataset, size) => {
  if (!Array.isArray(dataset.subsections) || dataset.subsections.length === 0) {
    return `
      ${plotEmbeds(dataset)}
      ${mediaNote(dataset.mediaNote)}
      ${comparisonList(normalizeComparisons(dataset), size, dataset.patchGrid)}
    `;
  }

  return `
    <div class="dataset-subsections">
      ${dataset.subsections
        .map(
          (subsection) => `
        <section class="dataset-subsection">
          <h3>${escapeHtml(subsection.title || "")}</h3>
          ${plotEmbeds(subsection)}
          ${mediaNote(subsection.mediaNote)}
          ${comparisonList(normalizeComparisons(subsection), size, subsection.patchGrid || dataset.patchGrid)}
        </section>
      `,
        )
        .join("")}
    </div>
  `;
};

const introTarget = document.querySelector("#intro-body");
const datasetNavTarget = document.querySelector("#dataset-nav");
const datasetTarget = document.querySelector("#datasets");
const introParagraphs = Array.isArray(siteData.intro)
  ? siteData.intro
  : [siteData.intro];
const datasets = Array.isArray(siteData.datasets) ? siteData.datasets : [];

document.querySelector("#site-title").textContent = siteData.title;
document.querySelector("#site-subtitle").textContent = siteData.subtitle;

introTarget.innerHTML = introParagraphs.map(renderIntroParagraph).join("");

datasetNavTarget.hidden = datasets.length < 2;
datasetNavTarget.innerHTML = datasets
  .map((dataset, index) => {
    const name = dataset.name || `Dataset ${index + 1}`;
    return `<a class="dataset-chip" href="#${slugify(name)}">${escapeHtml(name)}</a>`;
  })
  .join("");
if (!datasetNavTarget.hidden) {
  datasetNavTarget.insertAdjacentHTML(
    "afterbegin",
    `<p class="section-kicker">Datasets</p>`,
  );
}

datasetTarget.innerHTML = datasets.length
  ? datasets
      .map((dataset, index) => {
        const name = dataset.name || `Dataset ${index + 1}`;
        const size = dataset.size === "large" ? "large" : "small";
        return `
          <section class="dataset dataset-${size}" id="${slugify(name)}">
            <div class="dataset-heading">
              <h2>${escapeHtml(name)}</h2>
              ${
                dataset.description
                  ? `<p>${richText(dataset.description)}</p>`
                  : ""
              }
            </div>
            ${datasetBlocks(dataset, size)}
          </section>
        `;
      })
      .join("")
  : `<div class="empty-note">Add dataset entries in <code>main.js</code> to populate the page.</div>`;

const prepareVideo = (video) => {
  if (video.dataset.videoReady === "true") {
    return;
  }
  video.preload = "auto";
  video.load();
  video.dataset.videoReady = "true";
};

const playVideo = (video, options = {}) => {
  if (video.ended) {
    if (!options.restartEnded) {
      return;
    }
    video.currentTime = 0;
  }
  prepareVideo(video);
  video.play().catch(() => {});
};

const syncVideoGroup = (row) => {
  const videos = [...row.querySelectorAll("video[data-sync-video]")];
  if (videos.length < 2) {
    return;
  }

  let syncing = false;
  const syncTime = (source) => {
    if (syncing || !Number.isFinite(source.currentTime)) {
      return;
    }
    syncing = true;
    videos.forEach((video) => {
      if (video === source || !Number.isFinite(video.duration)) {
        return;
      }
      if (Math.abs(video.currentTime - source.currentTime) > 0.12) {
        video.currentTime = Math.min(source.currentTime, video.duration);
      }
    });
    syncing = false;
  };

  videos.forEach((video) => {
    video.addEventListener("play", () => {
      videos.forEach((other) => {
        if (other !== video && other.paused) {
          playVideo(other);
        }
      });
    });
    video.addEventListener("pause", () => {
      if (video.ended) {
        return;
      }
      videos.forEach((other) => {
        if (other !== video && !other.paused) {
          other.pause();
        }
      });
    });
    video.addEventListener("seeking", () => syncTime(video));
    video.addEventListener("timeupdate", () => syncTime(video));
  });
};

document.querySelectorAll(".comparison-row").forEach(syncVideoGroup);

const setupVideoStateControls = () => {
  document.querySelectorAll(".video-state-toggle").forEach((toggle) => {
    const video = toggle.closest(".media-card")?.querySelector("video[data-sync-video]");
    if (!video) {
      return;
    }

    const update = () => {
      const isPlaying = !video.paused && !video.ended;
      toggle.dataset.videoState = isPlaying ? "playing" : "paused";
      toggle.setAttribute(
        "aria-label",
        isPlaying ? "Pause generated video" : "Play generated video",
      );
    };

    toggle.addEventListener("click", () => {
      if (video.paused || video.ended) {
        if (video.ended) {
          video
            .closest(".comparison-row")
            ?.querySelectorAll("video[data-sync-video]")
            .forEach((rowVideo) => {
              rowVideo.currentTime = 0;
            });
        }
        playVideo(video, { restartEnded: true });
      } else {
        video.pause();
      }
    });
    video.addEventListener("play", update);
    video.addEventListener("pause", update);
    video.addEventListener("ended", update);
    update();
  });
};

const setupViewportAutoplay = () => {
  const rows = [...document.querySelectorAll(".comparison-row")];
  const playRow = (row) => {
    row.querySelectorAll("video[data-viewport-autoplay]").forEach(playVideo);
  };
  const pauseRow = (row) => {
    row.querySelectorAll("video[data-viewport-autoplay]").forEach((video) => {
      if (!video.paused) {
        video.pause();
      }
    });
  };

  if (!("IntersectionObserver" in window)) {
    rows.slice(0, 1).forEach(playRow);
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          playRow(entry.target);
        } else {
          pauseRow(entry.target);
        }
      });
    },
    {
      root: null,
      rootMargin: "120px 0px",
      threshold: 0.15,
    },
  );

  rows.forEach((row) => observer.observe(row));
};

setupVideoStateControls();
setupViewportAutoplay();

const zoomDialog = document.createElement("dialog");
zoomDialog.className = "image-zoom-dialog";
zoomDialog.innerHTML = `
  <button class="image-zoom-close" type="button" aria-label="Close zoom">Close</button>
  <p class="media-label zoom-title"></p>
  <img alt="">
  <div class="zoom-video-frame" hidden>
    <video controls playsinline></video>
    <svg class="patch-grid zoom-patch-grid" preserveAspectRatio="none" aria-hidden="true" hidden></svg>
  </div>
  <div class="series-labels zoom-series-labels"></div>
`;
document.body.append(zoomDialog);

const zoomImage = zoomDialog.querySelector("img");
const zoomVideoFrame = zoomDialog.querySelector(".zoom-video-frame");
const zoomVideo = zoomDialog.querySelector("video");
const zoomPatchGrid = zoomDialog.querySelector(".zoom-patch-grid");
const zoomTitle = zoomDialog.querySelector(".zoom-title");
const zoomLabels = zoomDialog.querySelector(".zoom-series-labels");
const resetZoomDialog = () => {
  zoomDialog.dataset.zoomMode = "";
  zoomTitle.textContent = "";
  zoomImage.hidden = true;
  zoomImage.removeAttribute("src");
  zoomImage.alt = "";
  zoomVideoFrame.hidden = true;
  zoomVideo.pause();
  zoomVideo.removeAttribute("src");
  zoomVideo.load();
  zoomPatchGrid.hidden = true;
  zoomPatchGrid.removeAttribute("viewBox");
  zoomPatchGrid.innerHTML = "";
  zoomLabels.hidden = true;
  zoomLabels.innerHTML = "";
};
resetZoomDialog();

document.querySelectorAll(".image-zoom-trigger").forEach((trigger) => {
  trigger.addEventListener("click", () => {
    resetZoomDialog();
    zoomTitle.textContent = trigger.dataset.zoomTitle || "";
    zoomImage.src = trigger.dataset.zoomSrc;
    zoomImage.alt = trigger.querySelector("img")?.alt || "";
    zoomImage.hidden = false;
    const labels = JSON.parse(trigger.dataset.zoomLabels || "[]");
    zoomLabels.hidden = labels.length === 0;
    zoomLabels.style.setProperty("--series-count", labels.length || 1);
    zoomLabels.innerHTML = labels
      .map((seriesLabel) => `<span>${escapeHtml(seriesLabel)}</span>`)
      .join("");
    zoomDialog.showModal();
  });
});
document.querySelectorAll(".video-zoom-trigger").forEach((trigger) => {
  trigger.addEventListener("click", () => {
    resetZoomDialog();
    zoomDialog.dataset.zoomMode = trigger.dataset.zoomSize || "small";
    zoomTitle.textContent = trigger.dataset.zoomTitle || "";
    zoomVideo.src = trigger.dataset.zoomSrc;
    zoomVideo.currentTime = 0;
    zoomVideoFrame.hidden = false;
    if (trigger.dataset.gridCols && trigger.dataset.gridRows) {
      const cols = Number(trigger.dataset.gridCols);
      const rows = Number(trigger.dataset.gridRows);
      zoomPatchGrid.setAttribute("viewBox", `0 0 ${cols} ${rows}`);
      zoomPatchGrid.innerHTML = gridLines(cols, rows);
      zoomPatchGrid.hidden = false;
    }
    zoomDialog.showModal();
    zoomVideo.play().catch(() => {});
  });
});
zoomDialog.querySelector(".image-zoom-close").addEventListener("click", () => {
  zoomDialog.close();
});
zoomDialog.addEventListener("click", (event) => {
  if (event.target === zoomDialog) {
    zoomDialog.close();
  }
});
zoomDialog.addEventListener("close", () => {
  resetZoomDialog();
});
