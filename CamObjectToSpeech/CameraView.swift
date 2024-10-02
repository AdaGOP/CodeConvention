//
//  CameraView.swift
//  CamObjectToSpeech
//
//  Created by khoirunnisa' rizky noor fatimah on 02/10/24.
//
import SwiftUI
import AVFoundation
import Vision

struct CameraView: UIViewControllerRepresentable {
    class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        var parent: CameraView
        let speechSynthesizer = AVSpeechSynthesizer() // Speech synthesizer
        
        init(parent: CameraView) {
            self.parent = parent
        }

        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            // Process the frame by calling the parent's processFrame method
            parent.processFrame(sampleBuffer, speechSynthesizer: speechSynthesizer)
        }
    }
    
    var session = AVCaptureSession()
    
    var model: VNCoreMLModel {
        let configuration = MLModelConfiguration()
        let mlModel = try! YOLOv3(configuration: configuration)
        return try! VNCoreMLModel(for: mlModel.model)
    }
    
    func makeUIViewController(context: Context) -> UIViewController {
        let controller = UIViewController()
        setupCameraSession(context: context) // Pass the context to setup method

        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.frame = controller.view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        controller.view.layer.addSublayer(previewLayer)
        
        
        self.session.startRunning()

        
        return controller
    }

    
    // Now takes context as a parameter to access the coordinator
    func setupCameraSession(context: Context) {
        session.beginConfiguration()
        
        // Setup video device
        let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
        let videoInput = try! AVCaptureDeviceInput(device: videoDevice)
        
        session.addInput(videoInput)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(context.coordinator, queue: DispatchQueue(label: "cameraQueue"))
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        
        session.commitConfiguration() // Commit the camera configuration
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Implement updates to the view controller if necessary
    }
    
    func processFrame(_ sampleBuffer: CMSampleBuffer, speechSynthesizer: AVSpeechSynthesizer) {
       
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                self.handleDetectedObjects(results, speechSynthesizer: speechSynthesizer)
            }
            
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try! handler.perform([request])
    }
    
    func handleDetectedObjects(_ results: [VNRecognizedObjectObservation], speechSynthesizer: AVSpeechSynthesizer) {
        for observation in results {
            // Use YOLO's labels
            let topLabelObservation = observation.labels.first!
            let label = topLabelObservation.identifier
            let confidence = topLabelObservation.confidence
            
            // Only announce if confidence is above a certain treshold
            if confidence > 0.5 {
                announceObject(label: label, speechSynthesizer: speechSynthesizer)
            }
        }
    }
    
    func announceObject(label: String, speechSynthesizer: AVSpeechSynthesizer) {
        let utterance = AVSpeechUtterance(string: "Detected: \(label)")
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speechSynthesizer.speak(utterance)
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }
}
