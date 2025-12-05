window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	bulmaCarousel.attach('#results-carousel', {
	  slidesToScroll: 1,
	  slidesToShow: 1,
	  loop: true,
	  infinite: true,
	  autoplay: true,
	  autoplaySpeed: 5000
	});
  
	bulmaSlider.attach();
  });

// Scroll-triggered animations
document.addEventListener('DOMContentLoaded', function() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, observerOptions);

  // Observe all sections
  document.querySelectorAll('.section').forEach(section => {
    section.classList.add('fade-in');
    observer.observe(section);
  });

  // Fallback for PDF images: if a browser can't render the PDF as an image,
  // automatically swap to the corresponding .jpg version.
  const pdfImages = document.querySelectorAll('img[src$=".pdf"]');
  pdfImages.forEach(img => {
    const pdfSrc = img.getAttribute('src');
    if (!pdfSrc) return;

    const jpgSrc = pdfSrc.replace(/\.pdf(\?.*)?$/i, '.jpg$1');

    const useJpg = () => {
      if (img.getAttribute('src') !== jpgSrc) {
        img.setAttribute('src', jpgSrc);
      }
    };

    // If loading the PDF fails, fall back to JPG.
    const onError = function() {
      useJpg();
      img.removeEventListener('error', onError);
    };

    img.addEventListener('error', onError);

    // If the image has already finished loading but failed, also swap.
    if (img.complete && img.naturalWidth === 0) {
      useJpg();
    }
  });
});

// Interactive t-SNE plot
function createTSNEPlot() {
  // Model name mapping and markers (matching Python code style)
  const MODEL_NAME_MAP = {
    'Runway Gen-4': 'Runway Gen4',
    'Wan2.1': 'Wan2.1',
    'Wan2.2': 'Wan2.2',
    'Opensora': 'Opensora',
    'Hunyuan': 'HunyuanVideo'
  };
  
  const MODEL_MARKERS = {
    'Runway Gen4': 'circle',
    'Wan2.1': 'square',
    'Wan2.2': 'diamond',
    'Opensora': 'triangle-up',
    'HunyuanVideo': 'triangle-down'
  };
  
  // Tab10 colormap colors (matching matplotlib)
  const TAB10_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  
  // Load JSON data
  fetch('static/images/tsne_gen_vs_real_centroids_no_train.json')
    .then(response => response.json())
    .then(data => {
      // Separate generated points and centroids
      const genPoints = data.filter(d => d.kind === 'gen');
      const centroids = data.filter(d => d.kind === 'centroid');
      
      // Get unique classes and models
      const classes = [...new Set(genPoints.map(d => d.class))].sort();
      const models = [...new Set(genPoints.map(d => d.model))].sort();
      
      // Create color map for classes
      const colorMap = {};
      classes.forEach((cls, i) => {
        colorMap[cls] = TAB10_COLORS[i % TAB10_COLORS.length];
      });
      
      // Create pretty model names and reverse mapping
      const modelToPretty = {};
      models.forEach(m => {
        modelToPretty[m] = MODEL_NAME_MAP[m] || m;
      });
      const uniquePrettyModels = [...new Set(Object.values(modelToPretty))].sort();
      
      // Reverse mapping: pretty name -> original model name
      const prettyToModel = {};
      Object.keys(modelToPretty).forEach(m => {
        prettyToModel[modelToPretty[m]] = m;
      });
      
      const traces = [];
      
      // 1) Generated samples - grouped by class and model
      classes.forEach(cls => {
        const clsPoints = genPoints.filter(d => d.class === cls);
        if (clsPoints.length === 0) return;
        
        uniquePrettyModels.forEach(pm => {
          const modelName = prettyToModel[pm];
          const modelPoints = clsPoints.filter(d => d.model === modelName);
          if (modelPoints.length === 0) return;
          
          const marker = MODEL_MARKERS[pm] || 'circle';
          
          traces.push({
            x: modelPoints.map(d => d.x),
            y: modelPoints.map(d => d.y),
            mode: 'markers',
            type: 'scatter',
            name: `${cls} (${pm})`,
            marker: {
              symbol: marker,
              size: 14,
              color: colorMap[cls],
              line: {
                color: 'black',
                width: 0.6
              },
              opacity: 0.6
            },
            showlegend: false,
            customdata: modelPoints.map(d => ({
              video_id: d.video_id,
              class: cls,
              model: pm
            })),
            hovertemplate: '<b>Class: %{customdata.class}</b><br>Model: %{customdata.model}<br>Video: %{customdata.video_id}<extra></extra>',
            class: cls,
            model: pm,
            visible: true
          });
        });
      });
      
      // 2) Centroids
      centroids.forEach(centroid => {
        traces.push({
          x: [centroid.x],
          y: [centroid.y],
          mode: 'markers',
          type: 'scatter',
          name: `Centroid: ${centroid.class}`,
          marker: {
            symbol: 'x',
            size: 25,
            color: 'white',
            line: {
              color: colorMap[centroid.class],
              width: 2.5
            }
          },
          showlegend: false,
          hovertemplate: '<b>Centroid: %{customdata.class}</b><extra></extra>',
          customdata: [{
            class: centroid.class,
            isCentroid: true
          }],
          class: centroid.class,
          isCentroid: true,
          visible: true
        });
      });
      
      // Combine all traces (no legend traces needed, using custom HTML legends)
      const allTraces = traces;
      
      // Calculate fixed axis ranges from all generated points (not centroids)
      const allGenX = genPoints.map(d => d.x);
      const allGenY = genPoints.map(d => d.y);
      const xMin = Math.min(...allGenX);
      const xMax = Math.max(...allGenX);
      const yMin = Math.min(...allGenY);
      const yMax = Math.max(...allGenY);
      
      // Add padding to ranges
      const xRange = xMax - xMin;
      const yRange = yMax - yMin;
      const xPadding = xRange * 0.05;
      const yPadding = yRange * 0.05;
      
      // Layout configuration matching matplotlib style
      const layout = {
        xaxis: {
          title: {
            text: 't-SNE dimension 1',
            font: { size: 14 }
          },
          showgrid: true,
          gridcolor: 'rgba(0,0,0,0.12)',
          showline: true,
          linecolor: 'black',
          mirror: false,
          zeroline: false,
          tickfont: { size: 12 },
          range: [xMin - xPadding, xMax + xPadding],
          fixedrange: true
        },
        yaxis: {
          title: {
            text: 't-SNE dimension 2',
            font: { size: 14 }
          },
          showgrid: true,
          gridcolor: 'rgba(0,0,0,0.12)',
          showline: true,
          linecolor: 'black',
          mirror: false,
          zeroline: false,
          tickfont: { size: 12 },
          range: [yMin - yPadding, yMax + yPadding],
          fixedrange: true
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        autosize: true,
        margin: { l: 60, r: 220, t: 60, b: 40 },
        showlegend: false, // Hide built-in legend, we'll use custom HTML legends
        hovermode: 'closest'
      };
      
      const config = {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
        responsive: true
      };
      
      // Store original traces and ranges for filtering
      window.tsneTraces = allTraces;
      window.tsneClasses = classes;
      window.tsneModels = uniquePrettyModels;
      window.tsneColorMap = colorMap;
      window.tsneModelMarkers = MODEL_MARKERS;
      const tsneRanges = {
        xMin: xMin - xPadding,
        xMax: xMax + xPadding,
        yMin: yMin - yPadding,
        yMax: yMax + yPadding
      };
      
      // Google Drive folder URL
      const GOOGLE_DRIVE_FOLDER = 'https://drive.google.com/drive/u/0/folders/1-VP6Rdr4qDNJ8k3IU2XcRbR85YIfqrRY';
      
      // Function to show video modal
      function showVideoModal(videoId, className, modelName) {
        const modal = document.getElementById('video-modal');
        const modalInfo = document.getElementById('video-modal-info');
        const modalIframe = document.getElementById('video-modal-iframe');
        
        if (!modal) return;
        
        // Set modal info
        modalInfo.textContent = `${className} - ${modelName}`;
        
        // Construct Google Drive embed URL
        // Google Drive requires file IDs for embedding, but we can try using the folder with search
        // The format for Google Drive file preview is: https://drive.google.com/file/d/FILE_ID/preview
        // Since we don't have file IDs, we'll use the folder search URL
        // Note: This may not work perfectly for embedding, but it's the best we can do without file IDs
        const searchTerm = encodeURIComponent(videoId);
        const embedUrl = `${GOOGLE_DRIVE_FOLDER}?q=${searchTerm}`;
        
        modalIframe.src = embedUrl;
        
        // Show modal
        modal.style.display = 'flex';
      }
      
      // Function to hide video modal
      function hideVideoModal() {
        const modal = document.getElementById('video-modal');
        const modalIframe = document.getElementById('video-modal-iframe');
        if (modal) {
          modal.style.display = 'none';
          // Clear iframe src to stop video playback
          if (modalIframe) {
            modalIframe.src = '';
          }
        }
      }
      
      // Close modal handlers
      document.addEventListener('DOMContentLoaded', function() {
        const closeBtn = document.getElementById('video-modal-close');
        const modal = document.getElementById('video-modal');
        
        if (closeBtn) {
          closeBtn.addEventListener('click', hideVideoModal);
        }
        
        if (modal) {
          modal.addEventListener('click', function(e) {
            if (e.target === modal) {
              hideVideoModal();
            }
          });
        }
        
        // Close on Escape key
        document.addEventListener('keydown', function(e) {
          if (e.key === 'Escape') {
            hideVideoModal();
          }
        });
      });
      
      // Plot
      Plotly.newPlot('tsne-plot', allTraces, layout, config).then(function() {
        // Add click handler to show video in modal
        const plotDiv = document.getElementById('tsne-plot');
        plotDiv.on('plotly_click', function(data) {
          if (data.points && data.points.length > 0) {
            const point = data.points[0];
            const customData = point.data.customdata[point.pointNumber];
            
            // Only show videos for generated points, not centroids
            if (customData && !customData.isCentroid && customData.video_id) {
              showVideoModal(customData.video_id, customData.class, customData.model);
            }
          }
        });
        
        // Add click handlers for filtering after plot is rendered
        let selectedClass = null;
        let selectedModel = null;
        
        // Store references for button updates
        let allClassButton = null;
        let allModelButtonRef = null;
        
        // Function to update plot visibility
        function updateVisibility() {
          const visibility = allTraces.map(trace => {
            if (trace.isCentroid) {
              return true; // Keep centroids visible
            }
            if (selectedModel === null && selectedClass === null) {
              return true; // Show all
            }
            if (selectedModel !== null && selectedClass !== null) {
              return trace.model === selectedModel && trace.class === selectedClass;
            }
            if (selectedModel !== null) {
              return trace.model === selectedModel;
            }
            if (selectedClass !== null) {
              return trace.class === selectedClass;
            }
            return true;
          });
          
          Plotly.restyle('tsne-plot', { visible: visibility }).then(function() {
            // Ensure ranges stay fixed after restyle
            Plotly.relayout('tsne-plot', {
              'xaxis.range': [tsneRanges.xMin, tsneRanges.xMax],
              'yaxis.range': [tsneRanges.yMin, tsneRanges.yMax],
              'xaxis.fixedrange': true,
              'yaxis.fixedrange': true
            });
          });
          
          // Update button states
          if (allClassButton) {
            if (selectedClass === null) {
              allClassButton.style.backgroundColor = 'rgba(0,0,0,0.1)';
            } else {
              allClassButton.style.backgroundColor = '';
            }
          }
          if (allModelButtonRef) {
            if (selectedModel === null) {
              allModelButtonRef.style.backgroundColor = 'rgba(0,0,0,0.1)';
            } else {
              allModelButtonRef.style.backgroundColor = '';
            }
          }
        }
        
        // Custom legends will be created after plot renders
        setTimeout(function() {
          // Legends are created below in the code
        }, 100);
        
        // Store update function for model legend
        window.tsneUpdateVisibility = updateVisibility;
        window.tsneGetSelectedClass = () => selectedClass;
        window.tsneGetSelectedModel = () => selectedModel;
        window.tsneSetSelectedClass = (cls) => { 
          selectedClass = cls; 
          updateVisibility();
        };
        window.tsneSetSelectedModel = (mod) => { 
          selectedModel = mod; 
          updateVisibility();
        };
        window.tsneAllModelButtonRef = () => allModelButtonRef;
      });
      
      // Create custom HTML legends for models (right) and centroid (top-left)
      const plotContainer = document.getElementById('tsne-plot');
      const plotDiv = plotContainer.querySelector('.plotly');
      
      // Model legend (right side) with click handlers
      const modelLegend = document.createElement('div');
      modelLegend.className = 'tsne-model-legend';
      modelLegend.style.cssText = 'position: absolute; right: 10px; top: 60px; background: rgba(255,255,255,0.85); border: 1px solid #666; padding: 8px; font-size: 12px; z-index: 1000; width: 160px; box-sizing: border-box;';
      modelLegend.innerHTML = '<div style="font-weight: bold; margin-bottom: 4px; font-size: 13px;">Generative models</div>';
      
      // Add "All" button for models
      const allModelButton = document.createElement('div');
      allModelButton.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px; font-weight: bold; background: rgba(0,0,0,0.1);';
      allModelButton.textContent = 'All';
      allModelButtonRef = allModelButton;
      allModelButton.addEventListener('click', function() {
        if (window.tsneSetSelectedModel) window.tsneSetSelectedModel(null);
        // Remove highlights from model items
        document.querySelectorAll('.tsne-model-legend > div[data-model]').forEach(el => {
          el.style.backgroundColor = '';
        });
      });
      allModelButton.addEventListener('mouseenter', function() {
        if (window.tsneGetSelectedModel && window.tsneGetSelectedModel() === null) {
          this.style.backgroundColor = 'rgba(0,0,0,0.15)';
        } else {
          this.style.backgroundColor = 'rgba(0,0,0,0.15)';
        }
      });
      allModelButton.addEventListener('mouseleave', function() {
        if (window.tsneGetSelectedModel && window.tsneGetSelectedModel() === null) {
          this.style.backgroundColor = 'rgba(0,0,0,0.1)';
        } else {
          this.style.backgroundColor = '';
        }
      });
      modelLegend.appendChild(allModelButton);
      
      uniquePrettyModels.forEach(pm => {
        const item = document.createElement('div');
        item.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px;';
        item.dataset.model = pm;
        item.addEventListener('click', function() {
          const currentSelectedModel = window.tsneGetSelectedModel ? window.tsneGetSelectedModel() : null;
          
          // Toggle model selection
          if (currentSelectedModel === pm) {
            if (window.tsneSetSelectedModel) window.tsneSetSelectedModel(null);
            item.style.backgroundColor = '';
          } else {
            if (window.tsneSetSelectedModel) window.tsneSetSelectedModel(pm);
            // Highlight selected and remove highlight from "All" and other items
            document.querySelectorAll('.tsne-model-legend > div[data-model]').forEach(el => {
              el.style.backgroundColor = '';
            });
            item.style.backgroundColor = 'rgba(0,0,0,0.1)';
          }
        });
        const marker = document.createElement('span');
        marker.style.cssText = `display: inline-block; width: 12px; height: 12px; margin-right: 6px; background: gray; border: 0.5px solid black; vertical-align: middle;`;
        // Set marker shape using CSS
        if (MODEL_MARKERS[pm] === 'circle') {
          marker.style.borderRadius = '50%';
        } else if (MODEL_MARKERS[pm] === 'square') {
          marker.style.borderRadius = '0';
        } else if (MODEL_MARKERS[pm] === 'diamond') {
          marker.style.transform = 'rotate(45deg)';
          marker.style.borderRadius = '0';
        } else if (MODEL_MARKERS[pm] === 'triangle-up') {
          // Create wrapper for triangle - same size as other markers for alignment
          marker.style.cssText = 'position: relative; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle;';
          // Black outline triangle (0.5px larger on each side = 1px total, matching 0.5px border)
          const outlineTriangle = document.createElement('span');
          outlineTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6.5px solid transparent; border-right: 6.5px solid transparent; border-bottom: 10.5px solid black;';
          // Gray fill triangle (0.5px smaller to show 0.5px border effect)
          const fillTriangle = document.createElement('span');
          fillTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid gray;';
          marker.appendChild(outlineTriangle);
          marker.appendChild(fillTriangle);
        } else if (MODEL_MARKERS[pm] === 'triangle-down') {
          // Create wrapper for triangle - same size as other markers for alignment
          marker.style.cssText = 'position: relative; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle;';
          // Black outline triangle (0.5px larger on each side = 1px total, matching 0.5px border)
          const outlineTriangle = document.createElement('span');
          outlineTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6.5px solid transparent; border-right: 6.5px solid transparent; border-top: 10.5px solid black;';
          // Gray fill triangle (0.5px smaller to show 0.5px border effect)
          const fillTriangle = document.createElement('span');
          fillTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 10px solid gray;';
          marker.appendChild(outlineTriangle);
          marker.appendChild(fillTriangle);
        }
        item.appendChild(marker);
        const label = document.createElement('span');
        label.textContent = pm;
        item.appendChild(label);
        modelLegend.appendChild(item);
      });
      plotContainer.appendChild(modelLegend);
      
      // Action classes legend (right side, below model legend)
      const classLegend = document.createElement('div');
      classLegend.className = 'tsne-class-legend';
      classLegend.style.cssText = 'position: absolute; right: 10px; top: 280px; background: rgba(255,255,255,0.85); border: 1px solid #666; padding: 8px; font-size: 12px; z-index: 1000; max-height: 400px; overflow-y: auto; width: 160px; box-sizing: border-box;';
      classLegend.innerHTML = '<div style="font-weight: bold; margin-bottom: 4px; font-size: 13px;">Action classes</div>';
      
      // Add "All" button for classes
      const allClassButtonHtml = document.createElement('div');
      allClassButtonHtml.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px; font-weight: bold; background: rgba(0,0,0,0.1);';
      allClassButtonHtml.textContent = 'All';
      allClassButtonHtml.addEventListener('click', function() {
        if (window.tsneSetSelectedClass) window.tsneSetSelectedClass(null);
        // Remove highlights from class items
        document.querySelectorAll('.tsne-class-legend > div[data-class]').forEach(el => {
          el.style.backgroundColor = '';
        });
      });
      allClassButtonHtml.addEventListener('mouseenter', function() {
        if (window.tsneGetSelectedClass && window.tsneGetSelectedClass() === null) {
          this.style.backgroundColor = 'rgba(0,0,0,0.15)';
        } else {
          this.style.backgroundColor = 'rgba(0,0,0,0.15)';
        }
      });
      allClassButtonHtml.addEventListener('mouseleave', function() {
        if (window.tsneGetSelectedClass && window.tsneGetSelectedClass() === null) {
          this.style.backgroundColor = 'rgba(0,0,0,0.1)';
        } else {
          this.style.backgroundColor = '';
        }
      });
      classLegend.appendChild(allClassButtonHtml);
      allClassButton = allClassButtonHtml;
      
      // Add class items
      classes.forEach((cls, i) => {
        const item = document.createElement('div');
        item.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px;';
        item.dataset.class = cls;
        item.addEventListener('click', function() {
          const currentSelectedClass = window.tsneGetSelectedClass ? window.tsneGetSelectedClass() : null;
          
          // Toggle class selection
          if (currentSelectedClass === cls) {
            if (window.tsneSetSelectedClass) window.tsneSetSelectedClass(null);
            item.style.backgroundColor = '';
          } else {
            if (window.tsneSetSelectedClass) window.tsneSetSelectedClass(cls);
            // Highlight selected and remove highlight from "All" and other items
            document.querySelectorAll('.tsne-class-legend > div[data-class]').forEach(el => {
              el.style.backgroundColor = '';
            });
            allClassButtonHtml.style.backgroundColor = '';
            item.style.backgroundColor = 'rgba(0,0,0,0.1)';
          }
        });
        const marker = document.createElement('span');
        marker.style.cssText = `display: inline-block; width: 12px; height: 12px; margin-right: 6px; background: ${colorMap[cls]}; border: 0.5px solid black; border-radius: 50%; opacity: 0.6;`;
        item.appendChild(marker);
        const label = document.createElement('span');
        label.textContent = cls;
        item.appendChild(label);
        classLegend.appendChild(item);
      });
      plotContainer.appendChild(classLegend);
      
      // Centroid legend (top-left) - use actual Plotly scatter point
      const representativeClassColor = classes.length > 0 ? colorMap[classes[0]] : '#666';
      const centroidLegend = document.createElement('div');
      centroidLegend.className = 'tsne-centroid-legend';
      centroidLegend.style.cssText = 'position: absolute; left: 10px; top: 60px; background: rgba(255,255,255,0.85); border: 1px solid #666; padding: 6px; font-size: 12px; z-index: 1000;';
      const centroidItem = document.createElement('div');
      centroidItem.style.cssText = 'display: flex; align-items: center;';
      
      // Create a small Plotly plot for the centroid marker
      const centroidMarkerDiv = document.createElement('div');
      centroidMarkerDiv.id = 'centroid-legend-marker';
      // Make container larger to ensure outline is fully visible
      centroidMarkerDiv.style.cssText = 'width: 24px; height: 24px; margin-right: 6px; display: inline-block; overflow: visible;';
      
      // Create a single scatter point matching the plot centroid style
      const centroidTrace = {
        x: [0],
        y: [0],
        mode: 'markers',
        type: 'scatter',
        marker: {
          symbol: 'x',
          size: 15,
          color: 'white',
          line: {
            color: 'gray',
            width: 2.5
          }
        },
        showlegend: false,
        hoverinfo: 'skip'
      };
      
      const centroidLayout = {
        xaxis: { visible: false, range: [-1, 1] },
        yaxis: { visible: false, range: [-1, 1] },
        margin: { l: 4, r: 4, t: 4, b: 4 },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
        autosize: true,
        width: 24,
        height: 24
      };
      
      const centroidConfig = {
        displayModeBar: false,
        staticPlot: true,
        responsive: false
      };
      
      Plotly.newPlot(centroidMarkerDiv, [centroidTrace], centroidLayout, centroidConfig);
      
      centroidItem.appendChild(centroidMarkerDiv);
      const centroidLabel = document.createElement('span');
      centroidLabel.textContent = 'Class centroid';
      centroidItem.appendChild(centroidLabel);
      centroidLegend.appendChild(centroidItem);
      plotContainer.appendChild(centroidLegend);
    })
    .catch(error => {
      console.error('Error loading t-SNE data:', error);
      document.getElementById('tsne-plot').innerHTML = '<p>Error loading plot data.</p>';
    });
}

// Initialize interactive ablation feature selection
function initAblationTable() {
  const container = document.getElementById('ablation-interactive');
  if (!container) return;
  
  // Baseline scores (all features included)
  const baselineScores = {
    consistency: 0.61,
    coherence: 0.64
  };
  
  // Scores when each feature is removed (only one at a time)
  const ablationScores = {
    'pose': { consistency: 0.56, coherence: 0.57 },
    'body-shape': { consistency: 0.54, coherence: 0.57 },
    'global-orientation': { consistency: 0.57, coherence: 0.57 },
    'keypoints': { consistency: 0.61, coherence: 0.57 },
    'visual-features': { consistency: 0.56, coherence: 0.59 },
    'motion': { consistency: 0.46, coherence: 0.50 }
  };
  
  // Feature labels
  const featureLabels = {
    'pose': 'Pose',
    'body-shape': 'Body shape',
    'global-orientation': 'Global orientation',
    'keypoints': 'Keypoints',
    'visual-features': 'Visual features',
    'motion': 'Motion'
  };
  
  const featureList = document.getElementById('feature-list');
  const selectedFeatures = new Set(['pose', 'body-shape', 'global-orientation', 'keypoints', 'visual-features', 'motion']);
  let removedFeature = null; // Track which single feature is removed
  const featureButtons = {};
  
  // Create feature buttons
  Object.keys(featureLabels).forEach(feature => {
    const button = document.createElement('button');
    button.className = 'feature-button';
    button.dataset.feature = feature;
    button.textContent = featureLabels[feature];
    button.style.cssText = `
      padding: 0.5rem 1rem;
      border: 1px solid rgba(0,0,0,0.15);
      background: white;
      color: #333;
      font-family: 'Noto Sans', sans-serif;
      font-weight: 300;
      font-size: 0.9rem;
      letter-spacing: 0.01em;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.2s ease;
    `;
    featureButtons[feature] = button;
    
    button.addEventListener('click', function() {
      if (removedFeature === feature) {
        // Restore this feature
        removedFeature = null;
        selectedFeatures.add(feature);
        this.style.background = 'white';
        this.style.color = '#333';
        this.style.textDecoration = 'none';
      } else {
        // Remove previous feature if any
        if (removedFeature !== null) {
          selectedFeatures.add(removedFeature);
          const prevButton = featureButtons[removedFeature];
          prevButton.style.background = 'white';
          prevButton.style.color = '#333';
          prevButton.style.textDecoration = 'none';
        }
        // Remove this feature
        removedFeature = feature;
        selectedFeatures.delete(feature);
        this.style.background = 'rgba(0,0,0,0.05)';
        this.style.color = '#999';
        this.style.textDecoration = 'line-through';
      }
      updateScores();
    });
    
    button.addEventListener('mouseenter', function() {
      if (removedFeature !== feature) {
        this.style.background = 'rgba(0,0,0,0.02)';
        this.style.borderColor = 'rgba(0,0,0,0.2)';
      }
    });
    
    button.addEventListener('mouseleave', function() {
      if (removedFeature !== feature) {
        this.style.background = 'white';
        this.style.borderColor = 'rgba(0,0,0,0.15)';
      }
    });
    
    featureList.appendChild(button);
  });
  
  // Update scores based on selected features
  function updateScores() {
    let consistency = baselineScores.consistency;
    let coherence = baselineScores.coherence;
    
    // If a feature is removed, use that ablation score
    if (removedFeature !== null) {
      consistency = ablationScores[removedFeature].consistency;
      coherence = ablationScores[removedFeature].coherence;
    }
    
    // Update display
    document.getElementById('score-consistency').textContent = consistency.toFixed(2);
    document.getElementById('score-coherence').textContent = coherence.toFixed(2);
    
    // Calculate and display drops
    const dropConsistency = baselineScores.consistency - consistency;
    const dropCoherence = baselineScores.coherence - coherence;
    
    const dropConsEl = document.getElementById('drop-consistency');
    const dropCohEl = document.getElementById('drop-coherence');
    
    if (removedFeature === null) {
      dropConsEl.textContent = '';
      dropCohEl.textContent = '';
    } else {
      dropConsEl.textContent = dropConsistency > 0 ? `↓ ${dropConsistency.toFixed(2)}` : '';
      dropCohEl.textContent = dropCoherence > 0 ? `↓ ${dropCoherence.toFixed(2)}` : '';
      dropConsEl.style.color = '#dc3545';
      dropCohEl.style.color = '#dc3545';
    }
  }
  
  // Initialize
  updateScores();
}

// Create Action Consistency vs Temporal Coherence plot
function createScoresPlot() {
  const plotElement = document.getElementById('scores-plot');
  if (!plotElement) {
    console.error('scores-plot element not found');
    return;
  }
  
  // Check if Plotly is loaded
  if (typeof Plotly === 'undefined') {
    console.error('Plotly is not loaded');
    plotElement.innerHTML = '<p style="color: red; padding: 2rem; text-align: center;">Error: Plotly library not loaded.</p>';
    return;
  }
  
  // Tab10 colormap colors (matching matplotlib)
  const TAB10_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];
  
  // Model name mapping
  const MODEL_NAME_MAP = {
    'Runway Gen-4': 'Runway Gen4',
    'Wan2.1': 'Wan2.1',
    'Wan2.2': 'Wan2.2',
    'Opensora': 'Opensora',
    'Hunyuan': 'HunyuanVideo'
  };
  
  // Google Drive folder URL
  const GOOGLE_DRIVE_FOLDER = 'https://drive.google.com/drive/u/0/folders/1-VP6Rdr4qDNJ8k3IU2XcRbR85YIfqrRY';
  
  // Function to show video modal with filename
  function showVideoModalByFilename(filename, className, modelName) {
    const modal = document.getElementById('video-modal');
    const modalInfo = document.getElementById('video-modal-info');
    const modalIframe = document.getElementById('video-modal-iframe');
    
    if (!modal) return;
    
    // Set modal info
    modalInfo.textContent = `${className} - ${modelName}`;
    
    // Construct Google Drive embed URL
    // Google Drive requires file IDs for proper embedding, but we can try using the folder with search
    // The format for Google Drive file preview is: https://drive.google.com/file/d/FILE_ID/preview
    // Since we don't have file IDs, we'll use the folder search URL
    const searchTerm = encodeURIComponent(filename);
    const embedUrl = `${GOOGLE_DRIVE_FOLDER}?q=${searchTerm}`;
    
    modalIframe.src = embedUrl;
    
    // Show modal
    modal.style.display = 'flex';
  }
  
  // Load scores data - try both relative and absolute paths
  const scoresPath = 'static/images/scores.json';
  fetch(scoresPath)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} for ${scoresPath}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('Scores data loaded successfully');
      // Parse data and extract class/model from filenames
      const points = [];
      const classes = new Set();
      const models = new Set();
      
      Object.keys(data).forEach(filename => {
        const scores = data[filename];
        // Parse filename: "Model_Class_XX_hash.mp4" or "Model_Number_Class_XX_hash.mp4"
        const parts = filename.replace('.mp4', '').split('_');
        let model = parts[0];
        // Get parts between model and the last two (number and hash)
        let classNameParts = parts.slice(1, -2);
        // Filter out purely numeric parts (like "768")
        classNameParts = classNameParts.filter(part => !/^\d+$/.test(part));
        let className = classNameParts.join('_');
        
        // Handle special cases - normalize model names from filenames
        if (model === 'RunwayGen4') {
          model = 'Runway Gen-4';
        } else if (model === 'Wan2p1') {
          model = 'Wan2.1';
        } else if (model === 'Wan2p2') {
          model = 'Wan2.2';
        }
        
        const prettyModel = MODEL_NAME_MAP[model] || model;
        
        points.push({
          x: scores.ac,
          y: scores.tc,
          filename: filename,
          model: model,
          prettyModel: prettyModel,
          class: className,
          ac: scores.ac,
          tc: scores.tc
        });
        
        classes.add(className);
        models.add(prettyModel);
      });
      
      const sortedClasses = Array.from(classes).sort();
      const sortedModels = Array.from(models).sort();
      
      // Create color map for classes
      const colorMap = {};
      sortedClasses.forEach((cls, i) => {
        colorMap[cls] = TAB10_COLORS[i % TAB10_COLORS.length];
      });
      
      // Model markers (same as t-SNE plot)
      const MODEL_MARKERS = {
        'Runway Gen4': 'circle',
        'Wan2.1': 'square',
        'Wan2.2': 'diamond',
        'Opensora': 'triangle-up',
        'HunyuanVideo': 'triangle-down'
      };
      
      // Calculate fixed axis ranges from all points
      const allX = points.map(p => p.x);
      const allY = points.map(p => p.y);
      const xMin = Math.min(...allX);
      const xMax = Math.max(...allX);
      const yMin = Math.min(...allY);
      const yMax = Math.max(...allY);
      
      // Add padding to ranges
      const xRange = xMax - xMin;
      const yRange = yMax - yMin;
      const xPadding = xRange * 0.05;
      const yPadding = yRange * 0.05;
      
      // Group points by class and model (same as t-SNE plot)
      const traces = [];
      sortedClasses.forEach(cls => {
        const classPoints = points.filter(p => p.class === cls);
        if (classPoints.length === 0) return;
        
        sortedModels.forEach(pm => {
          const modelPoints = classPoints.filter(p => p.prettyModel === pm);
          if (modelPoints.length === 0) return;
          
          const marker = MODEL_MARKERS[pm] || 'circle';
        
        traces.push({
            x: modelPoints.map(p => p.x),
            y: modelPoints.map(p => p.y),
          mode: 'markers',
          type: 'scatter',
            name: `${cls} (${pm})`,
          marker: {
              symbol: marker,
            size: 12,
            color: colorMap[cls],
            line: {
              color: 'black',
              width: 0.6
            },
            opacity: 0.6
          },
            showlegend: false,
            customdata: modelPoints.map(p => ({
            filename: p.filename,
            model: p.prettyModel,
            class: p.class,
            ac: p.ac,
            tc: p.tc
          })),
            hovertemplate: '<b>Class: %{customdata.class}</b><br>Model: %{customdata.model}<br>AC: %{x:.2f}<br>TC: %{y:.2f}<extra></extra>',
            class: cls,
            model: pm,
            visible: true
          });
        });
      });
      
      // Layout matching t-SNE plot style
      const layout = {
        xaxis: {
          title: {
            text: 'Action Consistency',
            font: { size: 14 }
          },
          showgrid: true,
          gridcolor: 'rgba(0,0,0,0.12)',
          showline: true,
          linecolor: 'black',
          mirror: false,
          zeroline: false,
          tickfont: { size: 12 },
          range: [xMin - xPadding, xMax + xPadding],
          fixedrange: true
        },
        yaxis: {
          title: {
            text: 'Temporal Coherence',
            font: { size: 14 }
          },
          showgrid: true,
          gridcolor: 'rgba(0,0,0,0.12)',
          showline: true,
          linecolor: 'black',
          mirror: false,
          zeroline: false,
          tickfont: { size: 12 },
          range: [yMin - yPadding, yMax + yPadding],
          fixedrange: true
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        autosize: true,
        margin: { l: 80, r: 220, t: 60, b: 60 },
        showlegend: false,
        hovermode: 'closest'
      };
      
      const config = {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
        responsive: true
      };
      
      // Store traces and ranges for filtering
      window.scoresTraces = traces;
      const scoresRanges = {
        xMin: xMin - xPadding,
        xMax: xMax + xPadding,
        yMin: yMin - yPadding,
        yMax: yMax + yPadding
      };
      
      // Create plot
      Plotly.newPlot('scores-plot', traces, layout, config).then(function() {
        // Add click handler to show video in modal
        const plotDiv = document.getElementById('scores-plot');
        if (!plotDiv) {
          console.error('scores-plot element not found');
          return;
        }
        
        plotDiv.on('plotly_click', function(data) {
          if (data.points && data.points.length > 0) {
            const point = data.points[0];
            const customData = point.data.customdata[point.pointNumber];
            
            if (customData && customData.filename) {
              showVideoModalByFilename(customData.filename, customData.class, customData.model);
            }
          }
        });
        
        // Filtering state
        let selectedClass = null;
        let selectedModel = null;
        let allClassButton = null;
        let allModelButtonRef = null;
        
        // Function to update plot visibility (same as t-SNE plot)
        function updateScoresVisibility() {
          const visibility = traces.map(trace => {
            if (selectedModel === null && selectedClass === null) {
              return true; // Show all
            }
            if (selectedModel !== null && selectedClass !== null) {
              return trace.model === selectedModel && trace.class === selectedClass;
            }
            if (selectedModel !== null) {
              return trace.model === selectedModel;
            }
            if (selectedClass !== null) {
              return trace.class === selectedClass;
            }
            return true;
          });
          
          Plotly.restyle('scores-plot', { visible: visibility }).then(function() {
            // Ensure ranges stay fixed after restyle
            Plotly.relayout('scores-plot', {
              'xaxis.range': [scoresRanges.xMin, scoresRanges.xMax],
              'yaxis.range': [scoresRanges.yMin, scoresRanges.yMax],
              'xaxis.fixedrange': true,
              'yaxis.fixedrange': true
            });
          });
          
          // Update button states
          if (allClassButton) {
            if (selectedClass === null) {
              allClassButton.style.backgroundColor = 'rgba(0,0,0,0.1)';
            } else {
              allClassButton.style.backgroundColor = '';
            }
          }
          if (allModelButtonRef) {
            if (selectedModel === null) {
              allModelButtonRef.style.backgroundColor = 'rgba(0,0,0,0.1)';
            } else {
              allModelButtonRef.style.backgroundColor = '';
            }
          }
        }
        
        // Store functions for legend access
        window.scoresSetSelectedClass = (cls) => {
          selectedClass = cls;
          updateScoresVisibility();
        };
        window.scoresGetSelectedClass = () => selectedClass;
        window.scoresSetSelectedModel = (mod) => {
          selectedModel = mod;
          updateScoresVisibility();
        };
        window.scoresGetSelectedModel = () => selectedModel;
        
        // Add legends (same as t-SNE plot)
        const plotContainer = document.getElementById('scores-plot');
        
        // Wait for plot to be fully rendered
        setTimeout(function() {
          // Model legend (right side) with click handlers
          const modelLegend = document.createElement('div');
          modelLegend.className = 'scores-model-legend';
          modelLegend.style.cssText = 'position: absolute; right: 10px; top: 60px; background: rgba(255,255,255,0.85); border: 1px solid #666; padding: 8px; font-size: 12px; z-index: 1000; width: 160px; box-sizing: border-box;';
          modelLegend.innerHTML = '<div style="font-weight: bold; margin-bottom: 4px; font-size: 13px;">Generative models</div>';
          
          // Add "All" button for models
          const allModelButton = document.createElement('div');
          allModelButton.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px; font-weight: bold; background: rgba(0,0,0,0.1);';
          allModelButton.textContent = 'All';
          allModelButtonRef = allModelButton;
          allModelButton.addEventListener('click', function() {
            if (window.scoresSetSelectedModel) window.scoresSetSelectedModel(null);
            // Remove highlights from model items
            document.querySelectorAll('.scores-model-legend > div[data-model]').forEach(el => {
              el.style.backgroundColor = '';
            });
          });
          allModelButton.addEventListener('mouseenter', function() {
            if (window.scoresGetSelectedModel && window.scoresGetSelectedModel() === null) {
              this.style.backgroundColor = 'rgba(0,0,0,0.15)';
            } else {
              this.style.backgroundColor = 'rgba(0,0,0,0.15)';
            }
          });
          allModelButton.addEventListener('mouseleave', function() {
            if (window.scoresGetSelectedModel && window.scoresGetSelectedModel() === null) {
              this.style.backgroundColor = 'rgba(0,0,0,0.1)';
            } else {
              this.style.backgroundColor = '';
            }
          });
          modelLegend.appendChild(allModelButton);
          
          sortedModels.forEach(pm => {
            const item = document.createElement('div');
            item.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px;';
            item.dataset.model = pm;
            item.addEventListener('click', function() {
              const currentSelectedModel = window.scoresGetSelectedModel ? window.scoresGetSelectedModel() : null;
              
              // Toggle model selection
              if (currentSelectedModel === pm) {
                if (window.scoresSetSelectedModel) window.scoresSetSelectedModel(null);
                item.style.backgroundColor = '';
              } else {
                if (window.scoresSetSelectedModel) window.scoresSetSelectedModel(pm);
                // Highlight selected and remove highlight from "All" and other items
                document.querySelectorAll('.scores-model-legend > div[data-model]').forEach(el => {
                  el.style.backgroundColor = '';
                });
                item.style.backgroundColor = 'rgba(0,0,0,0.1)';
              }
            });
            const marker = document.createElement('span');
            marker.style.cssText = `display: inline-block; width: 12px; height: 12px; margin-right: 6px; background: gray; border: 0.5px solid black; vertical-align: middle;`;
            // Set marker shape using CSS (same as t-SNE plot)
            if (MODEL_MARKERS[pm] === 'circle') {
              marker.style.borderRadius = '50%';
            } else if (MODEL_MARKERS[pm] === 'square') {
              marker.style.borderRadius = '0';
            } else if (MODEL_MARKERS[pm] === 'diamond') {
              marker.style.transform = 'rotate(45deg)';
              marker.style.borderRadius = '0';
            } else if (MODEL_MARKERS[pm] === 'triangle-up') {
              marker.style.cssText = 'position: relative; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle;';
              const outlineTriangle = document.createElement('span');
              outlineTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6.5px solid transparent; border-right: 6.5px solid transparent; border-bottom: 10.5px solid black;';
              const fillTriangle = document.createElement('span');
              fillTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid gray;';
              marker.appendChild(outlineTriangle);
              marker.appendChild(fillTriangle);
            } else if (MODEL_MARKERS[pm] === 'triangle-down') {
              marker.style.cssText = 'position: relative; display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle;';
              const outlineTriangle = document.createElement('span');
              outlineTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6.5px solid transparent; border-right: 6.5px solid transparent; border-top: 10.5px solid black;';
              const fillTriangle = document.createElement('span');
              fillTriangle.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 10px solid gray;';
              marker.appendChild(outlineTriangle);
              marker.appendChild(fillTriangle);
            }
            item.appendChild(marker);
            const label = document.createElement('span');
            label.textContent = pm;
            item.appendChild(label);
            modelLegend.appendChild(item);
          });
          plotContainer.appendChild(modelLegend);
          
          // Action classes legend (right side, below model legend)
          const classLegend = document.createElement('div');
          classLegend.className = 'scores-class-legend';
          classLegend.style.cssText = 'position: absolute; right: 10px; top: 280px; background: rgba(255,255,255,0.85); border: 1px solid #666; padding: 8px; font-size: 12px; z-index: 1000; max-height: 400px; overflow-y: auto; width: 160px; box-sizing: border-box;';
          classLegend.innerHTML = '<div style="font-weight: bold; margin-bottom: 4px; font-size: 13px;">Action classes</div>';
          
          // Add "All" button for classes
          const allClassButtonHtml = document.createElement('div');
          allClassButtonHtml.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px; font-weight: bold; background: rgba(0,0,0,0.1);';
          allClassButtonHtml.textContent = 'All';
          allClassButtonHtml.addEventListener('click', function() {
            if (window.scoresSetSelectedClass) window.scoresSetSelectedClass(null);
            // Remove highlights from class items
            document.querySelectorAll('.scores-class-legend > div[data-class]').forEach(el => {
              el.style.backgroundColor = '';
            });
          });
          allClassButtonHtml.addEventListener('mouseenter', function() {
            if (window.scoresGetSelectedClass && window.scoresGetSelectedClass() === null) {
              this.style.backgroundColor = 'rgba(0,0,0,0.15)';
            } else {
              this.style.backgroundColor = 'rgba(0,0,0,0.15)';
            }
          });
          allClassButtonHtml.addEventListener('mouseleave', function() {
            if (window.scoresGetSelectedClass && window.scoresGetSelectedClass() === null) {
              this.style.backgroundColor = 'rgba(0,0,0,0.1)';
            } else {
              this.style.backgroundColor = '';
            }
          });
          classLegend.appendChild(allClassButtonHtml);
          allClassButton = allClassButtonHtml;
          
          // Add class items
          sortedClasses.forEach((cls, i) => {
            const item = document.createElement('div');
            item.style.cssText = 'display: flex; align-items: center; margin: 3px 0; cursor: pointer; padding: 2px;';
            item.dataset.class = cls;
            item.addEventListener('click', function() {
              const currentSelectedClass = window.scoresGetSelectedClass ? window.scoresGetSelectedClass() : null;
              
              // Toggle class selection
              if (currentSelectedClass === cls) {
                if (window.scoresSetSelectedClass) window.scoresSetSelectedClass(null);
                item.style.backgroundColor = '';
              } else {
                if (window.scoresSetSelectedClass) window.scoresSetSelectedClass(cls);
                // Highlight selected and remove highlight from "All" and other items
                document.querySelectorAll('.scores-class-legend > div[data-class]').forEach(el => {
                  el.style.backgroundColor = '';
                });
                allClassButtonHtml.style.backgroundColor = '';
                item.style.backgroundColor = 'rgba(0,0,0,0.1)';
              }
            });
            
            const marker = document.createElement('span');
            marker.style.cssText = `display: inline-block; width: 12px; height: 12px; margin-right: 6px; background: ${colorMap[cls]}; border: 0.5px solid black; border-radius: 50%; opacity: 0.6;`;
            item.appendChild(marker);
            
            const label = document.createElement('span');
            label.textContent = cls;
            item.appendChild(label);
            
            classLegend.appendChild(item);
          });
          
          plotContainer.appendChild(classLegend);
        }, 100);
      });
    })
    .catch(error => {
      console.error('Error loading scores data:', error);
      const plotElement = document.getElementById('scores-plot');
      if (plotElement) {
        plotElement.innerHTML = '<p style="color: red; padding: 2rem; text-align: center;">Error loading plot data. Please check the console for details.</p>';
      }
    });
}

// Initialize plot when page loads
document.addEventListener('DOMContentLoaded', function() {
  if (document.getElementById('tsne-plot')) {
    createTSNEPlot();
    
    // Make plot responsive on window resize
    window.addEventListener('resize', function() {
      const plotDiv = document.getElementById('tsne-plot');
      if (plotDiv && plotDiv.querySelector('.plotly')) {
        Plotly.Plots.resize(plotDiv);
      }
    });
  }
  
  // Initialize ablation table
  initAblationTable();
  
  // Initialize scores plot - wait for element to be available
  function initScoresPlot() {
    const scoresPlotElement = document.getElementById('scores-plot');
    if (scoresPlotElement) {
      console.log('Initializing scores plot...');
      createScoresPlot();
      return true;
    }
    return false;
  }
  
  // Try to initialize immediately
  if (!initScoresPlot()) {
    // If element not found, wait and try again
    console.warn('scores-plot element not found, retrying...');
    let retries = 0;
    const maxRetries = 10;
    const retryInterval = setInterval(function() {
      retries++;
      if (initScoresPlot() || retries >= maxRetries) {
        clearInterval(retryInterval);
        if (retries >= maxRetries) {
          console.error('Failed to initialize scores plot after', maxRetries, 'retries');
        }
      }
    }, 200);
  }
  
  // Initialize citation copy button
  const copyBtn = document.getElementById('copy-citation-btn');
  const citationText = document.getElementById('citation-text');
  if (copyBtn && citationText) {
    copyBtn.addEventListener('click', function() {
      const text = citationText.textContent;
      navigator.clipboard.writeText(text).then(function() {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.style.background = 'rgba(0,0,0,0.05)';
        setTimeout(function() {
          copyBtn.textContent = originalText;
          copyBtn.style.background = 'white';
        }, 2000);
      }).catch(function(err) {
        console.error('Failed to copy:', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.opacity = '0';
        document.body.appendChild(textArea);
        textArea.select();
        try {
          document.execCommand('copy');
          copyBtn.textContent = 'Copied!';
          setTimeout(function() {
            copyBtn.textContent = 'Copy';
          }, 2000);
        } catch (err) {
          console.error('Fallback copy failed:', err);
        }
        document.body.removeChild(textArea);
      });
    });
  }
});