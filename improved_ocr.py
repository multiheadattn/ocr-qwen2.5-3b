from typing import List, Union, Optional
import torch
from PIL import Image
from pathlib import Path
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import argparse  # Added for command line argument parsing

class OCRTool:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 device=None):
        """
        Initialize the OCR tool.
        
        Args:
            model_name: The name of the pretrained model to use
            device: Computing device (cuda or cpu)
        """
        print(f"Loading model: {model_name}")
        start_time = time.time()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # using bitandbytes for 4 bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            quantization_config=bnb_config,
            device_map=self.device,
            low_cpu_mem_usage=True)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")


    def extract_text(self,
                     image_path: Union[str, Path]):
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        image = Image.open(image_path)
        prompt = "Please extract all text from this image. Show only the extracted text without any additional explanation."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        text_prompt = self.processor.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt = True)

        #process the image
        inputs = self.processor(
            text = [text_prompt],
            images = [image],
            padding =True,
            return_tensors = 'pt'
        )

        inputs = inputs.to(self.device)

        with torch.no_grad():
          output_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]):]
            for i in range(len(output_ids))
        ]

        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens = True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def extract_batch(self,
                      image_paths):
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            Dictionary mapping image paths to extracted text
        """
        results = {}
        for img_path in image_paths:
          print(f"Processing {img_path}...")
          results[img_path] = self.extract_text(img_path)
        return results

def main():
    """
    Main function to run the OCR tool from command line
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract text from images using OCR')
    parser.add_argument('-i', '--image', help='Path to a single image')
    parser.add_argument('-b', '--batch', nargs='+', help='Paths to multiple images for batch processing')
    args = parser.parse_args()

    # Initialize the OCR tool
    ocr_tool = OCRTool()

    # Check if image paths are provided via command line
    if args.image:
        # Single image mode
        extracted_text = ocr_tool.extract_text(image_path=args.image)
        print("\nExtracted Text:")
        print("-" * 40)
        print(extracted_text)
        print("-" * 40)
    elif args.batch:
        # Batch mode with command line arguments
        results = ocr_tool.extract_batch(args.batch)
        for img_path, text in results.items():
            print(f"\nExtracted Text from {img_path}:")
            print("-" * 40)
            print(text)
            print("-" * 40)

if __name__ == "__main__":
    main()