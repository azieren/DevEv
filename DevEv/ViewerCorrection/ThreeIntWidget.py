from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIntValidator

class ThreeEntryDialog(QDialog):
    def __init__(self, parent=None):
        super(ThreeEntryDialog, self).__init__(parent)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Three Entry Dialog')

        self.label1 = QLabel('Start:')
        self.entry1 = QLineEdit(self)
        self.entry1.setValidator(QIntValidator())

        self.label2 = QLabel('End:')
        self.entry2 = QLineEdit(self)
        self.entry2.setValidator(QIntValidator())

        self.label3 = QLabel('Step:')
        self.entry3 = QLineEdit(self)
        self.entry3.setValidator(QIntValidator())

        self.okButton = QPushButton('OK', self)
        self.okButton.clicked.connect(self.accept)

        self.cancelButton = QPushButton('Cancel', self)
        self.cancelButton.clicked.connect(self.reject)

        layout1 = QVBoxLayout()
        layout1.addWidget(self.label1)
        layout1.addWidget(self.entry1)
        layout2 = QVBoxLayout()
        layout2.addWidget(self.label2)
        layout2.addWidget(self.entry2)
        layout3 = QVBoxLayout()
        layout3.addWidget(self.label3)
        layout3.addWidget(self.entry3)
        
        entrieslayout = QHBoxLayout()
        entrieslayout.addLayout(layout1)
        entrieslayout.addLayout(layout2)
        entrieslayout.addLayout(layout3)
        
        layout4 = QHBoxLayout()
        layout4.addWidget(self.okButton)
        layout4.addWidget(self.cancelButton)
        
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(entrieslayout)
        mainLayout.addLayout(layout4)

        self.setLayout(mainLayout)

    def getInputs(self):
        start = int(self.entry1.text())
        end = int(self.entry2.text())
        step = int(self.entry3.text())

        return start, end, step

    def validateInputs(self):
        start = int(self.entry1.text())
        end = int(self.entry2.text())
        step = int(self.entry3.text())
        return 0 <= start <= end and end >= 0 and step >= 0

    def accept(self):
        if self.validateInputs():
            super().accept()
        else:
            # Show an error message or handle validation failure
            print("Start value must be larger or equal to End value.")
