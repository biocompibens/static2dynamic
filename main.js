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
      plot: "biotine/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "An in-house time lapse assay of human A549 lung cancer cells treated with biotine and imaged by confocal microscopy at the Curie Institute. Cells are marked with GFP for biotin effect, Rhodamin for membrane, and NucLight for nuclei.",
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
      plot: "chromalive/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "A time lapse assay from [Lippincott et al. 2025]{lippincott2025}. HeLa cells were exposed to ten staurosporine concentrations from 0 to 156.25 nM, processed with the Live Cell Painting assay (ChromaLIVE), and imaged every 30 minutes for six hours using spinning-disk confocal microscopy.",
      comparisons: [
        {
          gt: { src: "chromalive/display_vids/ground_truth_01_10s.mp4" },
          gen: { src: "chromalive/display_vids/generated_01_10s.mp4" },
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
            labels: ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase"],
          },
          gen: {
            src: "cell_cycle/gen_vids/extracted/trajectories_cell_0_0.mp4",
          },
        },
        {
          gt: {
            src: "cell_cycle/gt_vids_display/ground_truth_grid_02.png",
            labels: ["G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase"],
          },
          gen: {
            src: "cell_cycle/gen_vids/extracted/trajectories_cell_1_0.mp4",
          },
        },
      ],
    },
    {
      name: "Nocodazole and Docetaxel on MCF-7",
      size: "small",
      plots: [
        "nocodazole/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
        "docetaxel/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      ],
      description:
        "Subsets of the BBBC021 phenotypic profiling dataset from [Caie et al. 2010]{caie2010}, available from the Broad Bioimage Benchmark Collection ([Ljosa et al. 2012]{ljosa2012}). MCF-7 breast cancer cells are treated with compounds at eight concentrations and labeled for DNA, F-actin, and beta-tubulin. We use Nocodazole and Docetaxel independently.",
      comparisons: [],
    },
    {
      name: "Ependymal single-cell",
      size: "small",
      plot: "ependymal/3d_LDA_spline_projection_on_LDA_embedding_space_subsampled_10k.html",
      description:
        "Images from [Bankole et al. 2025]{bankole2025} on ependymal cells. Mouse lateral ependymal wall tissues were collected across postnatal stages, stained for cell junctions, centrioles, and deuterosomes, projected from 3D stacks to 2D ventricular surface images ([Shihavuddin et al. 2017]{shihavuddin2017}), then segmented into individual cell images.",
      comparisons: [],
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
      comparisons: [],
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
  escapeHtml(value).replace(
    /\[([^\]]+)\]\{([a-zA-Z0-9_-]+)\}/g,
    (match, label, id) => {
      const citation = citations[id];
      if (!citation?.url) {
        return label;
      }
      return `<a class="citation-link" href="${encodeURI(citation.url)}" target="_blank" rel="noreferrer">${escapeHtml(citation.label || label)}</a>`;
    },
  ).replace(/\*([^*]+)\*/g, "<em>$1</em>");

const renderIntroParagraph = (paragraph) => {
  const stages = paragraph.match(/^(.*?:)\s*1\.\s*(.*?)\s*2\.\s*(.*?)\s*3\.\s*(.*)$/);
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

const mediaCard = (item, label, datasetSize = "small") => {
  const source = encodeURI(item.src);
  const kind = mediaKind(item);
  const isImage = kind === "image";
  const isLargeDataset = datasetSize === "large";
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
  const media =
    isImage
      ? `<button class="image-zoom-trigger" type="button" data-zoom-src="${source}"${zoomTitle}${zoomLabels} aria-label="Zoom ${escapeHtml(label)}">
           <img src="${source}" alt="${escapeHtml(label)}" loading="lazy">
         </button>`
      : `<video controls autoplay playsinline muted preload="auto" data-sync-video>
           <source src="${source}">
           Your browser does not support the video tag.
         </video>
         <button class="video-zoom-trigger" type="button" data-zoom-src="${source}"${zoomTitle} data-zoom-size="${isLargeDataset ? "large" : "small"}" aria-label="Zoom ${escapeHtml(label)}">Zoom</button>`;

  return `
    <article class="media-card ${isImage ? "media-card-image" : "media-card-video"}">
      <p class="media-label">${escapeHtml(label)}</p>
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

const comparisonRow = (comparison, datasetSize) => {
  const gt = comparison.gt;
  const gen = comparison.gen;
  const gtLabel = gt && mediaKind(gt) === "video" ? "Ground truth" : "Ground truths";
  return `
    <article class="comparison-row">
      <div class="comparison-media">
        ${gt ? mediaCard(gt, gtLabel, datasetSize) : `<div class="empty-note">No ground truth sample.</div>`}
        ${gen ? mediaCard(gen, "Generated", datasetSize) : `<div class="empty-note">No generated sample.</div>`}
      </div>
    </article>
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

introTarget.innerHTML = introParagraphs
  .map(renderIntroParagraph)
  .join("");

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
        const comparisons = normalizeComparisons(dataset);
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
            ${plotEmbeds(dataset)}
            <div class="comparison-list">
              ${
                comparisons.length
                  ? comparisons.map((comparison) => comparisonRow(comparison, size)).join("")
                  : `<div class="empty-note">No media added yet.</div>`
              }
            </div>
          </section>
        `;
      })
      .join("")
  : `<div class="empty-note">Add dataset entries in <code>main.js</code> to populate the page.</div>`;

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
          other.play().catch(() => {});
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

const zoomDialog = document.createElement("dialog");
zoomDialog.className = "image-zoom-dialog";
zoomDialog.innerHTML = `
  <button class="image-zoom-close" type="button" aria-label="Close zoom">Close</button>
  <p class="media-label zoom-title"></p>
  <img alt="">
  <video controls playsinline></video>
  <div class="series-labels zoom-series-labels"></div>
`;
document.body.append(zoomDialog);

const zoomImage = zoomDialog.querySelector("img");
const zoomVideo = zoomDialog.querySelector("video");
const zoomTitle = zoomDialog.querySelector(".zoom-title");
const zoomLabels = zoomDialog.querySelector(".zoom-series-labels");
const resetZoomDialog = () => {
  zoomDialog.dataset.zoomMode = "";
  zoomTitle.textContent = "";
  zoomImage.hidden = true;
  zoomImage.removeAttribute("src");
  zoomImage.alt = "";
  zoomVideo.hidden = true;
  zoomVideo.pause();
  zoomVideo.removeAttribute("src");
  zoomVideo.load();
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
    zoomVideo.hidden = false;
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
