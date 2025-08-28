#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSlider>
#include <QLabel>
#include <QFileDialog>
#include <QCheckBox>
#include <QComboBox>
#include <QTimer>
#include <QImage>
#include <QPainter>
#include <QStatusBar>
#include <QMessageBox>
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>

class FluidDisplay : public QWidget {
    Q_OBJECT
    
public:
    explicit FluidDisplay(QWidget *parent = nullptr) : QWidget(parent) {
        setMinimumSize(600, 400);
        setBackgroundRole(QPalette::Base);
        setAutoFillBackground(true);
    }
    
    void setData(const float* density, const float* obstacles, 
                const float* vx, const float* vy,
                int width, int height, int frameIndex) {
        m_density = density;
        m_obstacles = obstacles;
        m_vx = vx;
        m_vy = vy;
        m_width = width;
        m_height = height;
        m_frameIndex = frameIndex;
        update();
    }
    
    void setDisplayMode(int mode) {
        m_displayMode = mode;
        update();
    }
    
    void setShowVectors(bool show) {
        m_showVectors = show;
        update();
    }
    
    void setVectorScale(float scale) {
        m_vectorScale = scale;
        update();
    }
    
    void setVectorSkip(int skip) {
        m_vectorSkip = skip;
        update();
    }
    
    QSize sizeHint() const override {
        return QSize(800, 600);
    }
    
protected:
    void paintEvent(QPaintEvent *event) override {
        Q_UNUSED(event);
        
        if (!m_density || m_width <= 0 || m_height <= 0) {
            QPainter painter(this);
            painter.setPen(Qt::black);
            painter.drawText(rect(), Qt::AlignCenter, "No data loaded");
            return;
        }
        
        // Create a QImage to draw on
        QImage image(m_width, m_height, QImage::Format_RGB32);
        
        // Draw the fluid data based on current mode
        if (m_displayMode == 0) { // Density
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    float value = m_density[global_at(x, y)];
                    // Map density to color: white -> green -> blue -> red
                    QRgb color;
                    
                    if (value <= 0.025f) {
                        // White to green
                        float t = value / 0.025f;
                        color = qRgb(255, static_cast<int>(255 * t), 0);
                    } else if (value <= 0.05f) {
                        // Green to blue
                        float t = (value - 0.025f) / 0.025f;
                        color = qRgb(0, 255 - static_cast<int>(255 * t), static_cast<int>(255 * t));
                    } else {
                        // Blue to red
                        float t = (value - 0.05f) / 0.05f;
                        color = qRgb(static_cast<int>(255 * t), 0, 255 - static_cast<int>(255 * t));
                    }
                    
                    // Cap at max value
                    if (value > 0.1f) color = qRgb(255, 0, 0);
                    
                    image.setPixel(x, y, color);
                }
            }
        } 
        else if (m_displayMode == 1 || m_displayMode == 2) { // Velocity X or Y
            const float* field = (m_displayMode == 1) ? m_vx : m_vy;
            float max_val = 0.0f;
            
            // Find max value for normalization
            for (int i = 0; i < m_width * m_height; i++) {
                max_val = std::max(max_val, std::abs(field[i]));
            }
            max_val = std::max(max_val, 1.0f); // Avoid division by zero
            
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    float value = field[global_at(x, y)];
                    float normalized = std::abs(value) / max_val;
                    
                    // Blue to red (negative to positive)
                    QRgb color;
                    if (value < 0) {
                        color = qRgb(0, 0, static_cast<int>(255 * normalized));
                    } else {
                        color = qRgb(static_cast<int>(255 * normalized), 0, 0);
                    }
                    
                    image.setPixel(x, y, color);
                }
            }
        }
        else { // Obstacles
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    float value = m_obstacles[global_at(x, y)];
                    QRgb color = (value > 0.5f) ? qRgb(100, 100, 100) : qRgb(255, 255, 255);
                    image.setPixel(x, y, color);
                }
            }
        }
        
        // Draw obstacles as semi-transparent overlay
        for (int y = 0; y < m_height; y++) {
            for (int x = 0; x < m_width; x++) {
                if (m_obstacles[global_at(x, y)] > 0.5f) {
                    QRgb color = image.pixel(x, y);
                    // Make it darker and more transparent
                    color = qRgb(qRed(color) * 0.7, qGreen(color) * 0.7, qBlue(color) * 0.7);
                    image.setPixel(x, y, color);
                }
            }
        }
        
        // Draw velocity vectors if enabled
        if (m_showVectors && (m_displayMode == 0 || m_displayMode == 3) && m_vx && m_vy) {
            QPainter painter(&image);
            painter.setRenderHint(QPainter::Antialiasing);
            QPen pen(Qt::white);
            pen.setWidth(1);
            painter.setPen(pen);
            
            for (int y = m_vectorSkip/2; y < m_height; y += m_vectorSkip) {
                for (int x = m_vectorSkip/2; x < m_width; x += m_vectorSkip) {
                    float vx = m_vx[global_at(x, y)];
                    float vy = m_vy[global_at(x, y)];
                    
                    // Skip very small vectors
                    if (std::abs(vx) < 0.1f && std::abs(vy) < 0.1f) continue;
                    
                    // Calculate vector end point
                    float length = std::sqrt(vx*vx + vy*vy);
                    float scale = m_vectorScale * length;
                    float end_x = x + vx * m_vectorScale;
                    float end_y = y + vy * m_vectorScale;
                    
                    // Draw vector
                    painter.drawLine(x, y, end_x, end_y);
                    
                    // Draw arrowhead
                    float angle = std::atan2(vy, vx);
                    float arrow_size = 3.0f;
                    QPointF points[3] = {
                        QPointF(end_x, end_y),
                        QPointF(end_x - arrow_size * std::cos(angle - M_PI/6), 
                                end_y - arrow_size * std::sin(angle - M_PI/6)),
                        QPointF(end_x - arrow_size * std::cos(angle + M_PI/6), 
                                end_y - arrow_size * std::sin(angle + M_PI/6))
                    };
                    painter.drawPolyline(points, 3);
                }
            }
        }
        
        // Draw frame number
        QPainter painter(&image);
        painter.setPen(Qt::white);
        painter.drawText(10, 20, QString("Frame: %1").arg(m_frameIndex));
        
        // Scale the image to fit the widget
        QImage scaled = image.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        
        // Draw on the widget
        QPainter widgetPainter(this);
        widgetPainter.fillRect(rect(), Qt::black);
        int x = (width() - scaled.width()) / 2;
        int y = (height() - scaled.height()) / 2;
        widgetPainter.drawImage(x, y, scaled);
    }
    
private:
    int global_at(int x, int y) const {
        return x + y * m_width;
    }
    
    const float* m_density = nullptr;
    const float* m_obstacles = nullptr;
    const float* m_vx = nullptr;
    const float* m_vy = nullptr;
    int m_width = 0;
    int m_height = 0;
    int m_frameIndex = 0;
    int m_displayMode = 0; // 0=density, 1=vx, 2=vy, 3=obstacles
    bool m_showVectors = true;
    float m_vectorScale = 0.05f;
    int m_vectorSkip = 8;
};

class FluidVisualizer : public QMainWindow {
    Q_OBJECT
    
public:
    FluidVisualizer(QWidget *parent = nullptr) : QMainWindow(parent) {
        setupUI();
        setWindowTitle("Fluid Simulation Visualizer");
        resize(1000, 700);
    }
    
private:
    void setupUI() {
        // Central widget and layout
        QWidget *centralWidget = new QWidget(this);
        QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
        
        // Fluid display
        m_display = new FluidDisplay(this);
        mainLayout->addWidget(m_display, 1);
        
        // Controls layout
        QHBoxLayout *controlsLayout = new QHBoxLayout();
        
        // Frame controls
        QVBoxLayout *frameLayout = new QVBoxLayout();
        m_frameSlider = new QSlider(Qt::Horizontal, this);
        m_frameSlider->setEnabled(false);
        connect(m_frameSlider, &QSlider::valueChanged, this, &FluidVisualizer::onFrameChanged);
        
        m_frameLabel = new QLabel("Frame: 0/0", this);
        frameLayout->addWidget(m_frameLabel);
        frameLayout->addWidget(m_frameSlider);
        
        QHBoxLayout *playbackLayout = new QHBoxLayout();
        m_prevButton = new QPushButton("Prev", this);
        m_prevButton->setEnabled(false);
        connect(m_prevButton, &QPushButton::clicked, this, &FluidVisualizer::prevFrame);
        
        m_playButton = new QPushButton("Play", this);
        m_playButton->setEnabled(false);
        connect(m_playButton, &QPushButton::clicked, this, &FluidVisualizer::togglePlayback);
        
        m_nextButton = new QPushButton("Next", this);
        m_nextButton->setEnabled(false);
        connect(m_nextButton, &QPushButton::clicked, this, &FluidVisualizer::nextFrame);
        
        playbackLayout->addWidget(m_prevButton);
        playbackLayout->addWidget(m_playButton);
        playbackLayout->addWidget(m_nextButton);
        frameLayout->addLayout(playbackLayout);
        
        controlsLayout->addLayout(frameLayout);
        
        // Display options
        QVBoxLayout *optionsLayout = new QVBoxLayout();
        
        QLabel *modeLabel = new QLabel("Display Mode:", this);
        m_modeCombo = new QComboBox(this);
        m_modeCombo->addItems({"Density", "Velocity X", "Velocity Y", "Obstacles"});
        m_modeCombo->setEnabled(false);
        connect(m_modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &FluidVisualizer::onDisplayModeChanged);
        
        m_vectorCheckBox = new QCheckBox("Show Velocity Vectors", this);
        m_vectorCheckBox->setChecked(true);
        m_vectorCheckBox->setEnabled(false);
        connect(m_vectorCheckBox, &QCheckBox::toggled, 
                this, &FluidVisualizer::onVectorToggled);
        
        optionsLayout->addWidget(modeLabel);
        optionsLayout->addWidget(m_modeCombo);
        optionsLayout->addWidget(m_vectorCheckBox);
        
        // Speed control
        QLabel *speedLabel = new QLabel("Playback Speed:", this);
        m_speedSlider = new QSlider(Qt::Horizontal, this);
        m_speedSlider->setRange(1, 100);
        m_speedSlider->setValue(30);
        m_speedSlider->setEnabled(false);
        connect(m_speedSlider, &QSlider::valueChanged, 
                this, &FluidVisualizer::onSpeedChanged);
        
        optionsLayout->addWidget(speedLabel);
        optionsLayout->addWidget(m_speedSlider);
        
        // Load button
        m_loadButton = new QPushButton("Load Simulation Data", this);
        connect(m_loadButton, &QPushButton::clicked, this, &FluidVisualizer::loadData);
        optionsLayout->addWidget(m_loadButton);
        
        controlsLayout->addLayout(optionsLayout);
        mainLayout->addLayout(controlsLayout);
        
        setCentralWidget(centralWidget);
        
        // Setup timer for animation
        m_timer = new QTimer(this);
        connect(m_timer, &QTimer::timeout, this, &FluidVisualizer::nextFrame);
    }
    
    void enableControls(bool enable) {
        m_frameSlider->setEnabled(enable);
        m_prevButton->setEnabled(enable);
        m_playButton->setEnabled(enable);
        m_nextButton->setEnabled(enable);
        m_modeCombo->setEnabled(enable);
        m_vectorCheckBox->setEnabled(enable);
        m_speedSlider->setEnabled(enable);
    }
    
    void loadData() {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Simulation Directory");
        if (dir.isEmpty()) return;
        
        // Try to load data files
        if (!loadDataFile(dir + "/data.bin", m_densityData) ||
            !loadDataFile(dir + "/obs.bin", m_obstaclesData) ||
            !loadDataFile(dir + "/v_x.bin", m_vxData) ||
            !loadDataFile(dir + "/v_y.bin", m_vyData)) {
            QMessageBox::critical(this, "Error", "Failed to load one or more data files");
            return;
        }
        
        // Determine grid dimensions and number of frames
        if (m_densityData.empty()) {
            QMessageBox::critical(this, "Error", "No density data loaded");
            return;
        }
        
        // We need to determine grid size by trial and error
        // The grid size is (width+2)*(height+2)
        // We'll assume the aspect ratio is reasonable (between 1:4 and 4:1)
        int totalElements = m_densityData.size() / sizeof(float);
        bool foundDimensions = false;
        
        for (int height = 10; height < 500; height++) {
            for (int width = 10; width < 1000; width++) {
                int gridElements = (width + 2) * (height + 2);
                if (totalElements % gridElements == 0) {
                    m_gridWidth = width;
                    m_gridHeight = height;
                    m_numFrames = totalElements / gridElements;
                    foundDimensions = true;
                    break;
                }
            }
            if (foundDimensions) break;
        }
        
        if (!foundDimensions) {
            QMessageBox::critical(this, "Error", "Could not determine grid dimensions from data");
            return;
        }
        
        statusBar()->showMessage(QString("Loaded %1 frames of %2x%3 grid")
                                .arg(m_numFrames).arg(m_gridWidth).arg(m_gridHeight));
        
        // Set up controls
        m_frameSlider->setRange(0, m_numFrames - 1);
        m_frameSlider->setEnabled(true);
        m_frameSlider->setValue(0);
        m_prevButton->setEnabled(true);
        m_playButton->setEnabled(true);
        m_nextButton->setEnabled(true);
        m_modeCombo->setEnabled(true);
        m_vectorCheckBox->setEnabled(true);
        m_speedSlider->setEnabled(true);
        
        updateFrameLabel(0);
        displayFrame(0);
    }
    
    bool loadDataFile(const QString& filename, std::vector<char>& data) {
        std::ifstream file(filename.toStdString(), std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            statusBar()->showMessage(QString("Could not open %1").arg(filename));
            return false;
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        data.resize(size);
        if (!file.read(data.data(), size)) {
            statusBar()->showMessage(QString("Error reading %1").arg(filename));
            return false;
        }
        
        return true;
    }
    
    void displayFrame(int frameIndex) {
        if (frameIndex < 0 || frameIndex >= m_numFrames) return;
        
        // Calculate the start position for this frame
        int frameSize = (m_gridWidth + 2) * (m_gridHeight + 2) * sizeof(float);
        int offset = frameIndex * frameSize / sizeof(float);
        
        m_display->setData(
            reinterpret_cast<float*>(m_densityData.data()) + offset,
            reinterpret_cast<float*>(m_obstaclesData.data()) + offset,
            reinterpret_cast<float*>(m_vxData.data()) + offset,
            reinterpret_cast<float*>(m_vyData.data()) + offset,
            m_gridWidth + 2,  // Include padding
            m_gridHeight + 2,
            frameIndex
        );
    }
    
    void updateFrameLabel(int frameIndex) {
        m_frameLabel->setText(QString("Frame: %1/%2").arg(frameIndex).arg(m_numFrames - 1));
    }
    
private slots:
    void onFrameChanged(int value) {
        displayFrame(value);
        updateFrameLabel(value);
    }
    
    void onDisplayModeChanged(int index) {
        m_display->setDisplayMode(index);
    }
    
    void onVectorToggled(bool checked) {
        m_display->setShowVectors(checked);
    }
    
    void onSpeedChanged(int value) {
        int interval = 100 - value + 1;
        m_timer->setInterval(interval);
    }
    
    void togglePlayback() {
        if (m_timer->isActive()) {
            m_timer->stop();
            m_playButton->setText("Play");
        } else {
            m_timer->start();
            m_playButton->setText("Pause");
        }
    }
    
    void nextFrame() {
        int next = (m_frameSlider->value() + 1) % m_numFrames;
        m_frameSlider->setValue(next);
    }
    
    void prevFrame() {
        int prev = (m_frameSlider->value() - 1 + m_numFrames) % m_numFrames;
        m_frameSlider->setValue(prev);
    }
    
private:
    FluidDisplay *m_display;
    QSlider *m_frameSlider;
    QLabel *m_frameLabel;
    QPushButton *m_prevButton;
    QPushButton *m_playButton;
    QPushButton *m_nextButton;
    QComboBox *m_modeCombo;
    QCheckBox *m_vectorCheckBox;
    QSlider *m_speedSlider;
    QPushButton *m_loadButton;
    QTimer *m_timer;
    
    std::vector<char> m_densityData;
    std::vector<char> m_obstaclesData;
    std::vector<char> m_vxData;
    std::vector<char> m_vyData;
    
    int m_gridWidth = 0;
    int m_gridHeight = 0;
    int m_numFrames = 0;
};

#include "fluid_visualizer.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    FluidVisualizer visualizer;
    visualizer.show();
    
    return app.exec();
}