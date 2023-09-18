package com.minimalapp.onnx;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;

public class MainActivity extends AppCompatActivity {

    TextView result;
    TextView duration;
    EditText input;
    ImageButton send;

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = Files.newOutputStream(file.toPath())) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e("FILE_MANAGER", assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        result = findViewById(R.id.result);
        duration = findViewById(R.id.duration);
        input = findViewById(R.id.input);
        send = findViewById(R.id.send);

        String model = assetFilePath(getBaseContext(),"model.onnx");
        String tokenizer = assetFilePath(getBaseContext(),"tokenizer.json");

        send.setOnClickListener(view -> {
            String inputComplete = input.getText().toString();

            if (!input.getText().toString().contains("[MASK]")){
                inputComplete = inputComplete + " [MASK]";
            }
            long start = System.currentTimeMillis();
            String resultModel = AlbertModel.inference(inputComplete, model, tokenizer);
            long end = System.currentTimeMillis();

            duration.setText(String.format("Inference Time: %d ms", end - start));
            result.setText(inputComplete.replace("[MASK]", resultModel.substring(1)));
        });
    }
}
