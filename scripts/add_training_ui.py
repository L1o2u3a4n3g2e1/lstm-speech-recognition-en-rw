"""
Add training UI tab to the existing HTML interface
This script enhances the web app with continuous training capabilities
"""

import re

def add_training_tab_to_html(html_file):
    """Add training tab to navigation and content areas"""

    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Training button for nav-tabs (add after History tab)
    training_nav_button = '''
                    <button class="nav-link" id="train-tab" data-bs-toggle="tab" data-bs-target="#train">
                        <i class="fas fa-brain"></i> Train Model
                    </button>'''

    # Find the History tab and add Training tab after it
    pattern = r'(<button class="nav-link" id="history-tab"[^<]*</button>)'
    content = re.sub(pattern, r'\1' + training_nav_button, content)

    # Training content tab (add after History content)
    training_content = '''
                <div class="tab-pane fade" id="train">
                    <div class="section-title">
                        <i class="fas fa-brain"></i> Fine-tune Model (Continuous Learning)
                    </div>

                    <p style="color: #666; margin-bottom: 1.5rem;">
                        Train the model on your own Kinyarwanda speech samples. The model will learn from your voice and adapt to your accent and speaking style.
                    </p>

                    <!-- Training Upload Area -->
                    <div class="upload-area" id="trainingUploadArea">
                        <div class="upload-icon">
                            <i class="fas fa-microphone"></i>
                        </div>
                        <div class="upload-text">
                            <strong>Upload Audio for Training</strong>
                        </div>
                        <div class="upload-text" style="font-size: 0.9rem;">
                            Drag & drop or click to select an audio file
                        </div>
                    </div>
                    <input type="file" id="trainingAudioInput" style="display: none;" accept=".wav,.mp3,.ogg,.m4a">

                    <!-- Transcript Input -->
                    <div class="mb-3">
                        <label for="trainingTranscript" class="form-label">Correct Transcript</label>
                        <textarea class="form-control" id="trainingTranscript" rows="3" placeholder="Type or paste the correct Kinyarwanda text for the audio you uploaded..."></textarea>
                    </div>

                    <!-- Learning Rate Slider -->
                    <div class="mb-3">
                        <label for="learningRate" class="form-label">Learning Rate (Advanced)</label>
                        <div class="input-group">
                            <input type="range" class="form-range" id="learningRate" min="0.00001" max="0.001" step="0.00001" value="0.0001">
                            <span style="margin-left: 1rem; font-weight: 600; min-width: 80px;" id="learningRateDisplay">0.0001</span>
                        </div>
                        <small class="text-muted">Lower = slower learning, Higher = faster but riskier</small>
                    </div>

                    <!-- Train Button -->
                    <button class="btn btn-primary-custom btn-custom" id="trainButton" style="width: 100%; margin-bottom: 1rem;">
                        <i class="fas fa-play"></i> Train on This Sample
                    </button>

                    <!-- Training Status -->
                    <div id="trainingStatus" class="result-box" style="display: none;">
                        <div class="result-label">Training Status</div>
                        <div class="result-text" id="trainingStatusText"></div>
                        <div style="margin-top: 1rem;">
                            <div class="d-flex justify-content-between" style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                                <span>Progress:</span>
                                <span id="samplesTrainedCount">0 samples</span>
                            </div>
                            <div style="background: #e9ecef; border-radius: 5px; overflow: hidden; height: 6px;">
                                <div id="trainingProgress" style="background: linear-gradient(90deg, #6f42c1, #764ba2); width: 0%; height: 100%; transition: width 0.3s;"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Loss Display -->
                    <div class="row" style="margin-top: 1.5rem;">
                        <div class="col-md-6">
                            <div class="card" style="border: 2px solid #f0f0f0;">
                                <div class="card-body">
                                    <div style="font-size: 0.9rem; color: #666;">Current Loss</div>
                                    <div style="font-size: 1.8rem; font-weight: 700; color: #6f42c1;" id="currentLoss">-</div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card" style="border: 2px solid #f0f0f0;">
                                <div class="card-body">
                                    <div style="font-size: 0.9rem; color: #666;">Samples Trained</div>
                                    <div style="font-size: 1.8rem; font-weight: 700; color: #20c997;" id="samplesTrained">0</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Save Model Button -->
                    <button class="btn btn-secondary-custom btn-custom" id="saveModelButton" style="width: 100%; margin-top: 1.5rem;">
                        <i class="fas fa-save"></i> Save Model Checkpoint
                    </button>

                    <!-- Instructions -->
                    <div class="alert alert-info" style="margin-top: 2rem;">
                        <h6 class="alert-heading"><i class="fas fa-info-circle"></i> How to Train the Model</h6>
                        <ol style="margin-bottom: 0; padding-left: 1.5rem;">
                            <li>Upload a clear Kinyarwanda speech recording (WAV, MP3, OGG, or M4A)</li>
                            <li>Type the correct transcription in the text area</li>
                            <li>Click "Train on This Sample" - the model will learn from your example</li>
                            <li>Repeat this 5-10 times with different sentences for better results</li>
                            <li>Click "Save Model Checkpoint" to preserve your trained model</li>
                        </ol>
                    </div>
                </div>'''

    # Find where to insert training content (after history content)
    pattern = r'(</div><!-- End History Tab -->)'
    if pattern not in content:
        # Alternative: find the last tab-pane and insert before closing
        pattern = r'(<div class="tab-pane fade" id="stats">.*?)</div><!-- End.*?Tab -->'
        replacement = r'\1</div><!-- End Stats Tab -->' + training_content + '\n'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        content = content.replace(pattern, r'\1' + training_content)

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ Training tab added to HTML interface")

def add_training_javascript(html_file):
    """Add JavaScript for training functionality"""

    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()

    training_js = '''

        // ===== TRAINING TAB FUNCTIONALITY =====

        // Learning rate display
        document.getElementById('learningRate')?.addEventListener('change', function() {
            document.getElementById('learningRateDisplay').textContent = this.value;
        });

        // Training upload area
        const trainingUploadArea = document.getElementById('trainingUploadArea');
        const trainingAudioInput = document.getElementById('trainingAudioInput');

        if (trainingUploadArea) {
            trainingUploadArea.addEventListener('click', () => trainingAudioInput.click());

            trainingUploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                trainingUploadArea.classList.add('dragover');
            });

            trainingUploadArea.addEventListener('dragleave', () => {
                trainingUploadArea.classList.remove('dragover');
            });

            trainingUploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                trainingUploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    trainingAudioInput.files = files;
                    document.querySelector('#trainingUploadArea .upload-text').innerHTML =
                        '<strong>' + files[0].name + ' selected</strong>';
                }
            });

            trainingAudioInput.addEventListener('change', () => {
                if (trainingAudioInput.files.length > 0) {
                    document.querySelector('#trainingUploadArea .upload-text').innerHTML =
                        '<strong>' + trainingAudioInput.files[0].name + ' selected</strong>';
                }
            });
        }

        // Train button
        document.getElementById('trainButton')?.addEventListener('click', async () => {
            const audioFile = trainingAudioInput.files[0];
            const transcript = document.getElementById('trainingTranscript').value.trim();
            const learningRate = parseFloat(document.getElementById('learningRate').value);

            if (!audioFile) {
                alert('Please select an audio file');
                return;
            }

            if (!transcript) {
                alert('Please provide the transcript');
                return;
            }

            const formData = new FormData();
            formData.append('audio', audioFile);
            formData.append('transcript', transcript);
            formData.append('learning_rate', learningRate);

            const trainButton = document.getElementById('trainButton');
            const originalText = trainButton.innerHTML;
            trainButton.disabled = true;
            trainButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';

            try {
                const response = await fetch('/api/train', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                const statusDiv = document.getElementById('trainingStatus');
                const statusText = document.getElementById('trainingStatusText');

                if (result.success) {
                    statusDiv.classList.remove('error');
                    statusDiv.classList.add('success', 'show');
                    statusText.innerHTML = `<strong>✓ Training successful!</strong><br>Loss: ${result.training_state.loss.toFixed(4)}`;

                    // Update stats
                    document.getElementById('currentLoss').textContent = result.training_state.loss.toFixed(4);
                    document.getElementById('samplesTrained').textContent = result.training_state.samples_trained;

                    // Clear transcript
                    document.getElementById('trainingTranscript').value = '';
                    trainingAudioInput.value = '';
                    document.querySelector('#trainingUploadArea .upload-text').innerHTML =
                        '<strong>Upload Audio for Training</strong>';
                } else {
                    statusDiv.classList.remove('success');
                    statusDiv.classList.add('error', 'show');
                    statusText.innerHTML = `<strong>✗ Training failed</strong><br>${result.error}`;
                }

                statusDiv.style.display = 'block';
            } catch (error) {
                const statusDiv = document.getElementById('trainingStatus');
                statusDiv.classList.remove('success');
                statusDiv.classList.add('error', 'show');
                statusDiv.style.display = 'block';
                document.getElementById('trainingStatusText').innerHTML =
                    `<strong>✗ Error:</strong> ${error.message}`;
            } finally {
                trainButton.disabled = false;
                trainButton.innerHTML = originalText;
            }
        });

        // Save model button
        document.getElementById('saveModelButton')?.addEventListener('click', async () => {
            const saveButton = document.getElementById('saveModelButton');
            const originalText = saveButton.innerHTML;
            saveButton.disabled = true;
            saveButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

            try {
                const response = await fetch('/api/model/save', { method: 'POST' });
                const result = await response.json();

                if (result.success) {
                    alert('✓ Model saved successfully!\\n' + result.message);
                } else {
                    alert('✗ Failed to save model:\\n' + result.message);
                }
            } catch (error) {
                alert('✗ Error saving model:\\n' + error.message);
            } finally {
                saveButton.disabled = false;
                saveButton.innerHTML = originalText;
            }
        });

        // Periodically update training status
        setInterval(async () => {
            try {
                const response = await fetch('/api/train/status');
                const result = await response.json();
                if (result.success) {
                    const state = result.training_state;
                    document.getElementById('currentLoss').textContent = state.loss.toFixed(4);
                    document.getElementById('samplesTrained').textContent = state.samples_trained;
                }
            } catch (e) {
                // Silent fail - endpoint may not exist yet
            }
        }, 5000);
    '''

    # Find the closing script tag and add before it
    if '</script>' in content:
        content = content.replace('</script>', training_js + '\n</script>')
    else:
        content += '\n<script>' + training_js + '</script>'

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ Training JavaScript added")

if __name__ == '__main__':
    html_file = '../web_app/templates/index.html'
    try:
        add_training_tab_to_html(html_file)
        add_training_javascript(html_file)
        print("\n✓ Web interface successfully enhanced with training capabilities!")
    except Exception as e:
        print(f"Error: {e}")
