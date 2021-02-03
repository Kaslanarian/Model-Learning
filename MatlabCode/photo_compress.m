function [] = photo_compress(photo_address, save_address, ratio, greycompress)
    % photo_address: 图片地址
    % save_address : 保存地址
    % ratio : 压缩的特征比例
    % greycompress : 为1表示彩转黑白
    if nargin == 3
        greycompress = 0;
    end

    img = double(imread(photo_address));

    if greycompress == 1 && size(img, 3) == 3
        img = double(rgb2gray(imread(photo_address)));
    end

    if size(img, 3) == 3
        R = img(:, :, 1);
        G = img(:, :, 2);
        B = img(:, :, 3);
        r = mySVD(R, ratio);
        g = mySVD(G, ratio);
        b = mySVD(B, ratio);
        compress_img = cat(3, r, g, b);
    else
        compress_img = mySVD(img, ratio);
    end

    imwrite(uint8(compress_img), save_address);
end
