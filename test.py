import sys
from PyQt5.QtWidgets import *

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Tabs Example")
window.resize(600, 400)

# Create tab widget
tabs = QTabWidget()

# Create tab pages
tab1 = QWidget()
tab2 = QWidget()
tab3 = QWidget()

# Add content to tab1
layout1 = QVBoxLayout()
layout1.addWidget(QLabel("This is Home Tab"))
layout1.addWidget(QPushButton("Home Button"))
tab1.setLayout(layout1)

# Add content to tab2
layout2 = QVBoxLayout()
layout2.addWidget(QLabel("Settings Tab"))
layout2.addWidget(QCheckBox("Enable Option"))
tab2.setLayout(layout2)

# Add content to tab3
layout3 = QVBoxLayout()
layout3.addWidget(QLabel("About Tab"))
tab3.setLayout(layout3)

# Add tabs
tabs.addTab(tab1, "Home")
tabs.addTab(tab2, "Settings")
tabs.addTab(tab3, "About")

# Main window layout
main_layout = QVBoxLayout()
main_layout.addWidget(tabs)

window.setLayout(main_layout)

window.show()

sys.exit(app.exec_())